# config.py
# Central settings file — every other module imports from here.
# Change values here instead of hunting through multiple files.

import os

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")        # put your PDFs here
INDEX_DIR  = os.path.join(BASE_DIR, "faiss_index") # FAISS saves here

# ── Chunking ─────────────────────────────────────────────────────
CHUNK_SIZE    = 800   # characters per chunk
CHUNK_OVERLAP = 150   # overlap between consecutive chunks

# ── Embedding model ───────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval config per query type ──────────────────────────────
RETRIEVAL_CONFIG = {
    "factual": {
        "top_k"   : 3,
        "strategy": "sparse",   # BM25 only
    },
    "analytical": {
        "top_k"   : 6,
        "strategy": "dense",    # FAISS only
    },
    "multi_hop": {
        "top_k"   : 8,
        "strategy": "hybrid",   # BM25 + FAISS combined
    },
}

# ── Re-ranker ─────────────────────────────────────────────────────
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_TOP_N = {
    "factual"   : 2,
    "analytical": 3,
    "multi_hop" : 4,
}

# ── LLM ──────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.1-8b-instant"
LLM_MAX_TOKENS = 512