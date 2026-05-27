"""
rag_pipeline.py

Coordinates document loading, index preparation, retrieval, and answer
generation for both the Streamlit app and the CLI entrypoint.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

from chunking import chunk_documents
from classifier import classify_query
from config import DATA_DIR
from env_utils import load_local_env
from generator import generate_answer
from loader import load_local_directory, load_uploaded_documents
from retriever import Retriever
from vector_store import DEFAULT_INDEX_PATH, DEFAULT_META_PATH, VectorStore

load_local_env()


def prepare_vector_store(
    data_mode: str = "existing",
    uploaded_files: Optional[Iterable[Any]] = None,
    append_uploaded: bool = False,
) -> Tuple[VectorStore, Dict[str, Any]]:
    """
    Build or load the vector store used by the app.

    Modes:
    - existing: load saved FAISS index if available, else build from ./data
    - upload: build from uploaded docs, optionally append them to saved index
    """
    mode = (data_mode or "existing").strip().lower()
    uploaded_files = list(uploaded_files or [])

    if mode == "existing":
        return _prepare_existing_store()
    if mode == "upload":
        return _prepare_uploaded_store(uploaded_files, append_uploaded=append_uploaded)
    raise ValueError(f"Unsupported data_mode: {data_mode}")


def answer_query(
    query: str,
    store: VectorStore,
    top_k: int = 5,
    use_expansion: bool = True,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run classification, retrieval, and generation for a single query.
    Returns the nested result structure consumed by app.py.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if store is None or not store.is_ready():
        raise ValueError("Vector store is not ready. Prepare a knowledge source first.")

    clean_query = query.strip()
    classification = classify_query(clean_query)

    retrieval_start = time.time()
    retriever = Retriever(store, top_k=max(1, int(top_k or 1)))
    retrieval = retriever.retrieve(clean_query, use_expansion=use_expansion)
    retrieval["latency_ms"] = round((time.time() - retrieval_start) * 1000, 2)
    retrieval["query_type"] = classification["type"]
    retrieval["strategy"] = classification["strategy"]
    retrieval["reason"] = classification["reason"]

    generation = generate_answer(
        query=clean_query,
        chunks=retrieval["chunks"],
        scores=retrieval["scores"],
        api_key=api_key,
    )

    total_latency_ms = round(float(retrieval["latency_ms"]) + float(generation["latency_ms"]), 2)

    return {
        "query": clean_query,
        "query_type": classification["type"],
        "retrieval": retrieval,
        "generation": generation,
        "sources": _collect_sources(retrieval["chunks"]),
        "total_latency_ms": total_latency_ms,
    }


def _prepare_existing_store() -> Tuple[VectorStore, Dict[str, Any]]:
    store = VectorStore()
    load_start = time.time()

    if store.load():
        build_info = {
            "mode": "existing",
            "data_label": DATA_DIR,
            "index_status": "loaded",
            "document_count": len({chunk.get("source", "?") for chunk in store.chunks}),
            "uploaded_document_count": 0,
            "chunk_count": store.total_chunks,
            "build_seconds": round(time.time() - load_start, 3),
        }
        return store, build_info

    docs = load_local_directory(DATA_DIR)
    if not docs:
        raise ValueError(
            f"No supported documents were found in '{DATA_DIR}'. Add PDF, TXT, or DOCX files first."
        )

    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("Documents were loaded, but no valid chunks were created from them.")

    build_info = store.build_from_chunks(chunks)
    store.save()
    build_info.update(
        {
            "mode": "existing",
            "data_label": DATA_DIR,
            "index_status": "built",
            "document_count": len(docs),
            "uploaded_document_count": 0,
            "chunk_count": store.total_chunks,
            "build_seconds": round(build_info.get("build_seconds", 0.0), 3),
        }
    )
    return store, build_info


def _prepare_uploaded_store(
    uploaded_files: List[Any],
    append_uploaded: bool = False,
) -> Tuple[VectorStore, Dict[str, Any]]:
    if not uploaded_files:
        raise ValueError("Upload at least one PDF, TXT, or DOCX file.")

    docs = load_uploaded_documents(uploaded_files)
    if not docs:
        raise ValueError("None of the uploaded files could be processed.")

    chunks = chunk_documents(docs)
    if not chunks:
        raise ValueError("Uploaded files were processed, but no valid chunks were created.")

    store = VectorStore()
    status = "built_from_upload"

    if append_uploaded:
        store.load()
        append_info = store.append_chunks(chunks)
        store.save(DEFAULT_INDEX_PATH, DEFAULT_META_PATH)
        build_info = {
            "mode": "upload",
            "data_label": f"{len(uploaded_files)} uploaded file(s) + saved index",
            "index_status": "appended",
            "document_count": len({chunk.get('source', '?') for chunk in store.chunks}),
            "uploaded_document_count": len(docs),
            "chunk_count": store.total_chunks,
            "build_seconds": round(append_info.get("build_seconds", 0.0), 3),
        }
        return store, build_info

    build_info = store.build_from_chunks(chunks)
    build_info.update(
        {
            "mode": "upload",
            "data_label": f"{len(uploaded_files)} uploaded file(s)",
            "index_status": status,
            "document_count": len(docs),
            "uploaded_document_count": len(docs),
            "chunk_count": store.total_chunks,
            "build_seconds": round(build_info.get("build_seconds", 0.0), 3),
        }
    )
    return store, build_info


def _collect_sources(chunks: List[Dict[str, Any]]) -> List[str]:
    seen = []
    for chunk in chunks:
        source = chunk.get("source", "Unknown source")
        if source not in seen:
            seen.append(source)
    return seen
