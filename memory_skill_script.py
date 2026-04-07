#!/usr/bin/env python3
"""
skill_memory_vector.py — Vector memory for agent (ChromaDB + X embeddings)

Indexes MEMORY.md and HISTORY.md in a persistent vector database (ChromaDB)
and answers semantic similarity search queries.
Embedding is delegated to X via the OpenAI-compatible API (text-embedding-bge-m3).

Usage:
  python skill_memory_vector.py index
  python skill_memory_vector.py search "natural language query" [--top-k 5]
  python skill_memory_vector.py stats

Storage: /root/.nanobot/workspace/chroma/ (in the nanobot-data volume)
X: https://api.X.ai/api/v1/embeddings
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap — install chromadb if missing (robust after container recreation)
# ---------------------------------------------------------------------------

def _ensure_deps() -> None:
    """Install chromadb if necessary."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        print("[bootstrap] Installing: chromadb", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "chromadb"]
        )

_ensure_deps()

import httpx  # noqa: E402  (already in the nanobot image)
import chromadb  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE = Path("/root/.nanobot/workspace")
CHROMA_DIR = WORKSPACE / "chroma"
MEMORY_FILE = WORKSPACE / "memory" / "MEMORY.md"
HISTORY_FILE = WORKSPACE / "memory" / "HISTORY.md"
COLLECTION_NAME = "agent_memory"

# X API (OpenAI-compatible)
X_API_URL = "https://api.X.ai/api/v1/embeddings"
EMBED_MODEL = "text-embedding-bge-m3"  # BGE-M3: excellent for multilingual


def _get_api_key() -> str:
    """Retrieve X API key from config.json or environment."""
    import json
    
    # First, try environment
    api_key = os.environ.get("X_API_KEY", "")
    if api_key:
        return api_key
    
    # Otherwise, read nanobot's config.json
    config_path = Path("/root/.nanobot/config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                api_key = config.get("providers", {}).get("custom", {}).get("apiKey", "")
                if api_key:
                    return api_key
        except (json.JSONDecodeError, IOError) as e:
            print(f"[skill_memory_vector] Error reading config.json: {e}", file=sys.stderr)
    
    return ""


X_API_KEY = _get_api_key()

# Max chunk size (in characters) for HISTORY.md
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# ---------------------------------------------------------------------------
# Embedding via X (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _embed(texts: list[str]) -> list[list[float]]:
    """Call X /embeddings API and return a list of embeddings."""
    if not texts:
        return []
    
    if not X_API_KEY:
        raise RuntimeError("X_API_KEY not defined in environment")
    
    try:
        resp = httpx.post(
            X_API_URL,
            headers={
                "Authorization": f"Bearer {X_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": EMBED_MODEL,
                "input": texts,
                "encoding_format": "float",
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible API returns {"data": [{"embedding": [...]}, ...]}
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
    except Exception as e:
        print(f"[ERROR] X embed failed: {e}", file=sys.stderr)
        raise


class _XEmbedFn(chromadb.EmbeddingFunction):
    """ChromaDB embedding function delegated to X."""

    def __call__(self, input: list[str]) -> list[list[float]]:
        return _embed(input)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks with overlap."""
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > size and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _split_memory_sections(text: str) -> list[tuple[str, str]]:
    """Split MEMORY.md by markdown section (## title)."""
    sections: list[tuple[str, str]] = []
    current_title = "general"
    current_body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))
    return [(t, b) for t, b in sections if b]


def _split_history_entries(text: str) -> list[str]:
    """Split HISTORY.md by timestamped entry [YYYY-MM-DD ...]."""
    entries = re.split(r"(?=$$\d{4}-\d{2}-\d{2})", text.strip())
    result: list[str] = []
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # if entry is too long, split into chunks
        if len(entry) > CHUNK_SIZE:
            result.extend(_chunk_text(entry))
        else:
            result.append(entry)
    return result


# ---------------------------------------------------------------------------
# ChromaDB Client
# ---------------------------------------------------------------------------

def _get_collection() -> chromadb.Collection:
    """Return ChromaDB collection with X embedding function."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_XEmbedFn(),
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_index() -> None:
    """Index MEMORY.md and HISTORY.md in ChromaDB, then consolidate HISTORY.md (keep 24h).
    
    Output rule: SILENT on stdout (no messages to User).
    All progress logs go to stderr (diagnostic only).
    Only concrete errors raise an exception (captured by main).
    """
    from datetime import datetime, timedelta
    
    collection = _get_collection()
    
    documents: list[str] = []
    ids: list[str] = []
    metadatas: list[dict] = []
    
    # --- MEMORY.md ---
    if MEMORY_FILE.exists():
        text = MEMORY_FILE.read_text(encoding="utf-8")
        for i, (title, body) in enumerate(_split_memory_sections(text)):
            if not body.strip():
                continue
            documents.append(f"[MEMORY — {title}]\n{body}")
            ids.append(f"memory_{i}")
            metadatas.append({"source": "MEMORY.md", "section": title})
        print(f"  MEMORY.md: {len([d for d in ids if d.startswith('memory_')])} sections indexed", file=sys.stderr)
    else:
        print("  MEMORY.md not found — ignored", file=sys.stderr)
    
    # --- HISTORY.md ---
    history_entries_count = 0
    if HISTORY_FILE.exists():
        text = HISTORY_FILE.read_text(encoding="utf-8")
        entries = _split_history_entries(text)
        for i, entry in enumerate(entries):
            documents.append(f"[HISTORY]\n{entry}")
            ids.append(f"history_{i}")
            metadatas.append({"source": "HISTORY.md", "entry_idx": i})
        history_entries_count = len(entries)
        print(f"  HISTORY.md: {history_entries_count} entries indexed", file=sys.stderr)
    else:
        print("  HISTORY.md not found — ignored", file=sys.stderr)
    
    if not documents:
        print("  No documents to index.", file=sys.stderr)
        return
    
    # Upsert in batches of 50 (limited by API request size)
    batch = 50
    total = len(documents)
    for start in range(0, total, batch):
        end = start + batch
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]
        embeddings = _embed(batch_docs)
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=embeddings,
        )
        print(f"  batch {start+1}-{min(end, total)}/{total} indexed", file=sys.stderr)
    
    print(f"  Index updated — {total} chunks in {CHROMA_DIR}", file=sys.stderr)
    print(f"  Embedding model: {EMBED_MODEL} (X API)", file=sys.stderr)
    
    # --- Consolidation: keep only the last 24 hours in HISTORY.md ---
    if HISTORY_FILE.exists():
        try:
            text = HISTORY_FILE.read_text(encoding="utf-8")
            lines = text.strip().split("\n")
            
            cutoff_time = datetime.now() - timedelta(hours=24)
            recent_lines = []
            current_entry = []
            keep_entry = False
            
            for line in lines:
                # Detect the start of a new entry [YYYY-MM-DD HH:MM:SS]
                if line.startswith("[") and "]" in line:
                    # Save previous entry if it's recent
                    if current_entry and keep_entry:
                        recent_lines.extend(current_entry)
                    
                    # New entry
                    current_entry = [line]
                    # Extract timestamp
                    try:
                        timestamp_str = line.split("]")[0][1:]  # [YYYY-MM-DD HH:MM:SS
                        entry_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        keep_entry = entry_time >= cutoff_time
                    except (ValueError, IndexError):
                        # If no valid timestamp, keep the entry
                        keep_entry = True
                else:
                    current_entry.append(line)
            
            # Last entry
            if current_entry and keep_entry:
                recent_lines.extend(current_entry)
            
            # Write only recent entries
            if len(recent_lines) < len(lines):
                HISTORY_FILE.write_text("\n".join(recent_lines) + "\n", encoding="utf-8")
                removed = len(lines) - len(recent_lines)
                print(f"  Consolidation: {removed} lines archived (kept 24h)", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: error consolidating HISTORY.md: {e}", file=sys.stderr)
    
    # Intentionally empty stdout output — no message to User


def cmd_search(query: str, top_k: int = 5) -> None:
    """Search by semantic similarity."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        print("⚠️  Empty index — first run: python skill_memory_vector.py index")
        return

    query_embedding = _embed([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    print(f"\n🔍 Search: « {query} »\n{'-' * 60}")
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        score = round(1 - dist, 3)  # cosine distance → similarity
        source = meta.get("source", "?")
        section = meta.get("section", "")
        label = f"{source}" + (f" — {section}" if section else "")
        print(f"\n[{i}] score={score}  source={label}")
        print(doc[:400] + ("…" if len(doc) > 400 else ""))

    print(f"\n{'-' * 60}")


def cmd_stats() -> None:
    """Display vector database stats."""
    if not CHROMA_DIR.exists():
        print("Vector database not initialized. Run: python skill_memory_vector.py index")
        return
    collection = _get_collection()
    count = collection.count()
    size = sum(f.stat().st_size for f in CHROMA_DIR.rglob("*") if f.is_file())
    print(f"Collection    : {COLLECTION_NAME}")
    print(f"Chunks        : {count}")
    print(f"DB size       : {size // 1024} KB")
    print(f"Directory     : {CHROMA_DIR}")
    print(f"Embed model   : {EMBED_MODEL} (X API)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vector memory ChromaDB + X for agent"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("index", help="Index MEMORY.md and HISTORY.md")

    search_p = subparsers.add_parser("search", help="Search by similarity")
    search_p.add_argument("query", help="Natural language query")
    search_p.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")

    subparsers.add_parser("stats", help="Vector database statistics")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index()
    elif args.command == "search":
        cmd_search(args.query, args.top_k)
    elif args.command == "stats":
        cmd_stats()


if __name__ == "__main__":
    main()