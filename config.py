"""
config.py — Константи, шляхи та змінні середовища.
"""

import os
from pathlib import Path

# ── Шляхи ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
STATIC_DIR     = BASE_DIR / "static"
KNOWLEDGE_DIR  = BASE_DIR / "knowledge_base"
UPLOADS_DIR    = BASE_DIR / "uploads"
CHROMA_DIR     = BASE_DIR / "chroma_db"
STATES_FILE    = CHROMA_DIR / "index_states.json"   # mtime-кеш на диску
SUMMARIES_FILE = CHROMA_DIR / "doc_summaries.json"  # стислі огляди документів

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",         "http://localhost:11434")
MODEL        = os.getenv("OLLAMA_MODEL",        "qwen3:14b")
EMBED_MODEL  = os.getenv("OLLAMA_EMBED_MODEL",  "nomic-embed-text")

# ── RAG / чанкінг ────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "600"))   # символів на чанк
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))    # перекриття між чанками
TOP_K         = int(os.getenv("TOP_K",         "6"))     # кількість чанків для retrieval
