import time

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
import re

from classifier import classify_query, print_classification


def build_bm25_index(chunks: list) -> tuple:
    """Build a BM25 index from a list of chunk documents."""
    if not chunks:
        raise ValueError("No chunks provided to build BM25 index.")

    print(f"  Building BM25 index from {len(chunks)} chunks...")
    tokenized_corpus = [re.findall(r"\w+", chunk.page_content.lower()) for chunk in chunks]

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"  ✓ BM25 index built")
    return bm25, chunks


def retrieve_sparse(query: str, bm25: BM25Okapi, chunks: list, top_k: int) -> list:
    """Run BM25 keyword-based retrieval."""
    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    return [chunks[i] for i in top_indices]


def retrieve_dense(query: str, vector_store: FAISS, top_k: int) -> list:
    """Run dense retrieval against the FAISS index."""
    return vector_store.similarity_search(query, k=top_k)


def retrieve_hybrid(
    query: str,
    bm25: BM25Okapi,
    chunks: list,
    vector_store: FAISS,
    top_k: int,
) -> list:
    """Combine sparse and dense retrieval, then deduplicate results."""
    half_k = top_k // 2
    rest_k = top_k - half_k

    sparse_docs = retrieve_sparse(query, bm25, chunks, top_k=half_k)
    dense_docs = retrieve_dense(query, vector_store, top_k=rest_k)

    seen = set()
    combined = []

    for doc in sparse_docs + dense_docs:
        key = doc.page_content[:100].strip()
        if key not in seen:
            combined.append(doc)
            seen.add(key)

    return combined[:top_k]


def adaptive_retrieve(
    query: str,
    vector_store: FAISS,
    bm25: BM25Okapi,
    chunks: list,
    verbose: bool = True,
) -> tuple:
    """Classify a query, pick a strategy, and retrieve matching chunks."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    start = time.time()
    query = expand_query(query)
    clf_result = classify_query(query)
    top_k = clf_result["top_k"]
    strategy = clf_result["strategy"]

    if verbose:
        print_classification(query, clf_result)

    if strategy == "sparse":
        docs = retrieve_sparse(query, bm25, chunks, top_k)
    elif strategy == "dense":
        docs = retrieve_dense(query, vector_store, top_k)
    elif strategy == "hybrid":
        docs = retrieve_hybrid(query, bm25, chunks, vector_store, top_k)
    else:
        print(f"  Unknown strategy '{strategy}' — using dense")
        docs = retrieve_dense(query, vector_store, top_k)

    elapsed = round((time.time() - start) * 1000, 2)

    if verbose:
        print(f"\n  Retrieved : {len(docs)} chunks via {strategy} strategy ({elapsed} ms)")

    return docs, clf_result


def expand_query(query: str) -> str:
    """Expand a few common short forms to improve retrieval recall."""
    query = query.lower()
    expansions = {
        "nlp": "natural language processing nlp",
        "ml": "machine learning ml",
        "ai": "artificial intelligence ai",
    }

    for key, value in expansions.items():
        if key in query:
            return value

    return query

def compare_baseline_vs_adaptive(
    queries: list,
    vector_store: FAISS,
    bm25: BM25Okapi,
    chunks: list,
) -> list:
    """Compare a fixed dense baseline against adaptive retrieval."""
    results = []

    for query in queries:
        t0 = time.time()
        baseline_docs = retrieve_dense(query, vector_store, top_k=5)
        baseline_ms = round((time.time() - t0) * 1000, 2)

        t0 = time.time()
        adaptive_docs, clf = adaptive_retrieve(
            query, vector_store, bm25, chunks, verbose=False,
        )
        adaptive_ms = round((time.time() - t0) * 1000, 2)

        results.append({
            "query": query,
            "query_type": clf["type"],
            "strategy": clf["strategy"],
            "baseline_docs": len(baseline_docs),
            "adaptive_docs": len(adaptive_docs),
            "baseline_ms": baseline_ms,
            "adaptive_ms": adaptive_ms,
        })

    return results


def print_comparison_table(results: list) -> None:
    """Print a compact baseline-vs-adaptive comparison table."""
    print("\n" + "=" * 80)
    print(
        f"{'Query':<35} {'Type':<12} {'Strategy':<10} "
        f"{'Base docs':>9} {'Apt docs':>9} "
        f"{'Base ms':>8} {'Apt ms':>8}"
    )
    print("─" * 80)

    for r in results:
        q = r["query"][:33] + ".." if len(r["query"]) > 33 else r["query"]
        print(
            f"{q:<35} {r['query_type']:<12} {r['strategy']:<10} "
            f"{r['baseline_docs']:>9} {r['adaptive_docs']:>9} "
            f"{r['baseline_ms']:>7}ms {r['adaptive_ms']:>7}ms"
        )

    avg_base_docs = sum(r["baseline_docs"] for r in results) / len(results)
    avg_apt_docs = sum(r["adaptive_docs"] for r in results) / len(results)
    avg_base_ms = sum(r["baseline_ms"] for r in results) / len(results)
    avg_apt_ms = sum(r["adaptive_ms"] for r in results) / len(results)
    doc_reduction = round((1 - avg_apt_docs / avg_base_docs) * 100)

    print("─" * 80)
    print(
        f"\n  Avg docs sent to LLM  — "
        f"Baseline: {avg_base_docs:.1f}   "
        f"Adaptive: {avg_apt_docs:.1f}   "
        f"({doc_reduction}% reduction)"
    )
    print(
        f"  Avg retrieval latency — "
        f"Baseline: {avg_base_ms:.1f}ms  "
        f"Adaptive: {avg_apt_ms:.1f}ms"
    )
    print("=" * 80)
