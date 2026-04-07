name	memory-vector
description	Vector memory ChromaDB — indexing and semantic search in MEMORY.md and HISTORY.md

Skill — Vector Memory (ChromaDB)
This skill enables searching past memories through semantic similarity rather than simply reading the entire MEMORY.md and HISTORY.md files.

Script
skills/skill_memory_vector/skill_memory_vector.py

Dependencies (pre-installed in the image): chromadb, httpx
Embedding model: text-embedding-bge-m3 (VeniceAI API)
Database storage: /root/.nanobot/workspace/chroma/

Available commands
Index memory
python skills/skill_memory_vector/skill_memory_vector.py index
Reads memory/MEMORY.md (by ## sections) and memory/HISTORY.md (by timestamped entries)
Creates or updates embeddings in ChromaDB
Should be executed after each important memory consolidation

Search by similarity
python skills/skill_memory_vector/skill_memory_vector.py search "topic or question" [--top-k 5]
Returns the top-k chunks closest to the query (cosine similarity score)
--top-k: number of results (default: 5)

Statistics
python skills/skill_memory_vector/skill_memory_vector.py stats
Displays the number of indexed chunks and database size

When to use this skill
Automatic indexing (systemd timer): indexing is automatically managed by the nanobot-index.timer systemd timer on the host. No manual action required in HEARTBEAT.md.

Contextual search: before answering a question that involves remembering preferences, past decisions, or personal context:

python skills/skill_memory_vector/skill_memory_vector.py search "user's preferences for Word reports" --top-k 3
python skills/skill_memory_vector/skill_memory_vector.py search "XXX project planning"
python skills/skill_memory_vector/skill_memory_vector.py search "error fixed last week"
Golden rule: if the answer depends on the past and the current context is insufficient → first search in the vector memory before reading the entire MEMORY.md.

Technical behavior
Aspect	Detail
MEMORY.md splitting	By ## Title sections
HISTORY.md splitting	By [YYYY-MM-DD ...] entries, 800 char chunks if too long
Embedding	text-embedding-bge-m3 via VeniceAI API
Similarity	Cosine distance (score 0→1, 1 = identical)
Persistence	/root/.nanobot/workspace/chroma/ (nanobot-data volume)
Upsert	Idempotent — re-indexing does not duplicate data

Recommended workflow
New session → [work] → Memory consolidation
    → python skills/skill_memory_vector.py index
    → ready for next searches
