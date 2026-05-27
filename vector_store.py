"""
vector_store.py — FAISS-based vector store
Supports dynamic index creation from uploaded docs and loading a prebuilt index.
"""

import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import faiss
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL = EMBEDDING_MODEL
DEFAULT_INDEX_PATH = "faiss_index/index.faiss"
DEFAULT_META_PATH  = "faiss_index/metadata.pkl"
LEGACY_META_PATH   = "faiss_index/index.pkl"


class VectorStore:
    """
    Wraps a FAISS flat-L2 index with chunk metadata.
    All similarity scores are converted to cosine-like [0,1] values.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        local_only = os.getenv("EMBEDDING_LOCAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}
        try:
            self.model = SentenceTransformer(model_name, local_files_only=local_only)
        except Exception as exc:
            mode = "local cache only" if local_only else "local cache or download"
            raise RuntimeError(
                f"Could not load embedding model '{model_name}' using {mode}. "
                "If this is your first run, connect once so Hugging Face can cache the model, "
                "or disable EMBEDDING_LOCAL_ONLY after the model is downloaded. "
                f"Original error: {exc}"
            ) from exc
        self.index: Optional[faiss.IndexFlatIP] = None   # inner-product on normed vecs = cosine
        self.chunks: List[dict] = []                     # parallel list of chunk metadata
        self.embed_dim: int = self.model.get_sentence_embedding_dimension()

    def _normalise_chunk(self, chunk: Any, chunk_id: int) -> Dict[str, Any]:
        if isinstance(chunk, dict):
            metadata = dict(chunk.get("metadata", {}))
            text = chunk.get("text", "")
            source = chunk.get("source") or metadata.get("source") or f"Chunk {chunk_id + 1}"
        elif isinstance(chunk, Document):
            metadata = dict(chunk.metadata or {})
            text = chunk.page_content
            source = metadata.get("source") or metadata.get("filename") or f"Chunk {chunk_id + 1}"
        else:
            metadata = {}
            text = getattr(chunk, "page_content", str(chunk))
            source = f"Chunk {chunk_id + 1}"

        metadata.setdefault("source", source)
        metadata["chunk_id"] = chunk_id

        return {
            "text": text,
            "source": source,
            "chunk_id": chunk_id,
            "metadata": metadata,
        }

    def _normalise_chunks(self, chunks: List[Any], start_index: int = 0) -> List[Dict[str, Any]]:
        return [
            self._normalise_chunk(chunk, start_index + offset)
            for offset, chunk in enumerate(chunks)
        ]

    # ── Build ─────────────────────────────────────────────────────────────────

    def build_from_chunks(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Embed all chunks and build a new FAISS index.
        Returns summary stats for the build.
        """
        normalised_chunks = self._normalise_chunks(chunks)
        if not normalised_chunks:
            raise ValueError("No chunks available to build the vector store.")

        t0 = time.time()
        texts = [c["text"] for c in normalised_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        self.index = faiss.IndexFlatIP(self.embed_dim)   # inner product on unit vecs = cosine
        self.index.add(embeddings)
        self.chunks = normalised_chunks

        sources = {chunk["source"] for chunk in normalised_chunks}
        return {
            "build_seconds": time.time() - t0,
            "chunk_count": len(normalised_chunks),
            "document_count": len(sources),
        }

    def append_chunks(self, chunks: List[Any]) -> Dict[str, Any]:
        """
        Add new chunks to the current index. Builds a new index if needed.
        """
        if self.index is None or self.index.ntotal == 0:
            return self.build_from_chunks(chunks)

        normalised_chunks = self._normalise_chunks(chunks, start_index=len(self.chunks))
        if not normalised_chunks:
            raise ValueError("No chunks available to append to the vector store.")

        t0 = time.time()
        texts = [c["text"] for c in normalised_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        self.index.add(embeddings)
        self.chunks.extend(normalised_chunks)

        sources = {chunk["source"] for chunk in normalised_chunks}
        return {
            "build_seconds": time.time() - t0,
            "chunk_count": len(normalised_chunks),
            "document_count": len(sources),
        }

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, index_path: str = DEFAULT_INDEX_PATH, meta_path: str = DEFAULT_META_PATH):
        if self.index is None:
            raise ValueError("Cannot save an empty vector store.")
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def _load_legacy_chunks(self, legacy_meta_path: str) -> List[Dict[str, Any]]:
        with open(legacy_meta_path, "rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, tuple) or len(payload) != 2:
            return []

        docstore, index_to_docstore_id = payload
        doc_map = getattr(docstore, "_dict", {})
        chunks = []

        for chunk_idx, doc_id in sorted(index_to_docstore_id.items()):
            doc = doc_map.get(doc_id)
            if doc is None:
                continue
            chunks.append(self._normalise_chunk(doc, int(chunk_idx)))

        return chunks

    def load(
        self,
        index_path: str = DEFAULT_INDEX_PATH,
        meta_path: str = DEFAULT_META_PATH,
        legacy_meta_path: str = LEGACY_META_PATH,
    ) -> bool:
        """Load a prebuilt index. Returns True on success."""
        if not os.path.exists(index_path):
            return False

        self.index = faiss.read_index(index_path)

        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                self.chunks = pickle.load(f)
            return True

        if os.path.exists(legacy_meta_path):
            try:
                self.chunks = self._load_legacy_chunks(legacy_meta_path)
            except Exception:
                self.chunks = []
            return bool(self.chunks)

        return False

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> Tuple[List[dict], List[float], float]:
        """
        Search for the top-k most similar chunks.

        Returns:
            chunks      — list of chunk dicts (text, source, chunk_id)
            scores      — cosine similarity scores [0, 1]
            latency_ms  — retrieval time in milliseconds
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Build or load an index first.")

        t0 = time.time()
        q_emb = self.model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb, dtype="float32")

        k = min(top_k, self.index.ntotal)
        scores_raw, indices = self.index.search(q_emb, k)

        latency_ms = (time.time() - t0) * 1000

        retrieved_chunks = []
        scores = []
        for idx, score in zip(indices[0], scores_raw[0]):
            if idx == -1:
                continue
            retrieved_chunks.append(self.chunks[idx])
            # cosine similarity on unit vectors is already in [-1, 1]; clip to [0, 1]
            scores.append(float(np.clip(score, 0.0, 1.0)))

        return retrieved_chunks, scores, latency_ms

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    def is_ready(self) -> bool:
        return self.index is not None and self.index.ntotal > 0 and bool(self.chunks)
