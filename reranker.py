# reranker.py
# Responsibility: Score retrieved chunks for precise relevance
#                 and filter out low-quality results before
#                 they reach the LLM generator.
#
# Why this matters:
#   FAISS/BM25 retrieval is approximate.
#   The cross-encoder re-reads (query + doc) together —
#   much more accurate than separate embedding comparison.
#
# Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - Size    : ~25MB (downloads once, cached locally)
#   - Speed   : fast on CPU (~50ms for 8 pairs)
#   - Quality : strong for passage relevance scoring

import time
from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, RERANKER_TOP_N


# ─────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────

def load_reranker() -> CrossEncoder:
    """
    Loads the cross-encoder model.

    First run  : downloads ~25MB from HuggingFace, then caches.
    Later runs : loads from local cache instantly.

    Returns:
        CrossEncoder model object
    """
    print(f"  Loading re-ranker model: {RERANKER_MODEL}")
    print(f"  (First run downloads ~25MB — subsequent runs instant)")

    reranker = CrossEncoder(
        model_name = RERANKER_MODEL,
        max_length = 512    # max tokens per (query + doc) pair
    )

    print(f"  ✓ Re-ranker loaded")
    return reranker


# ─────────────────────────────────────────────────────────────────
# CORE RE-RANKING FUNCTION
# ─────────────────────────────────────────────────────────────────

def rerank(
    query     : str,
    docs      : list,
    reranker  : CrossEncoder,
    query_type: str = "analytical",
    verbose   : bool = True
) -> list:
    """
    Re-scores retrieved documents using a cross-encoder
    and returns only the top-N most relevant ones.

    How it works:
      1. Build (query, doc_text) pairs — one per chunk
      2. Feed all pairs to the cross-encoder in one batch
      3. Get a relevance score for each pair
      4. Sort by score (highest = most relevant)
      5. Keep only top_n based on query type
      6. Return filtered, ordered list

    Args:
        query      : the original user question
        docs       : list of Document objects from retrieval
        reranker   : loaded CrossEncoder model
        query_type : 'factual' | 'analytical' | 'multi_hop'
                     determines how many docs to keep
        verbose    : if True, prints scoring table

    Returns:
        list of top-N Document objects, best first
    """
    if not docs:
        print("  No documents to re-rank.")
        return []

    if len(docs) == 1:
        # Nothing to compare — return as-is
        return docs

    # How many docs to keep after re-ranking
    # (defined per query type in config.py)
    top_n = RERANKER_TOP_N.get(query_type, 3)
    top_n = min(top_n, len(docs))   # can't keep more than we have

    start = time.time()

    # ── Step 1: Build (query, doc) pairs ─────────────────────────
    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    # ── Step 2: Score all pairs in one batch ─────────────────────
    # show_progress_bar=False keeps output clean
    scores = reranker.predict(pairs, show_progress_bar=False)

    # ── Step 3: Zip scores with docs ─────────────────────────────
    scored_docs = list(zip(scores, docs))

    # ── Step 4: Sort highest score first ─────────────────────────
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    elapsed = round((time.time() - start) * 1000, 2)

    # ── Step 5: Print scoring table if verbose ───────────────────
    if verbose:
        print(f"\n  Re-ranking {len(docs)} chunks "
              f"→ keeping top {top_n}  ({elapsed} ms)")
        print(f"\n  {'Rank':<5} {'Score':>7}  {'Status':<6}  "
              f"Preview (first 65 chars)")
        print(f"  {'─'*5} {'─'*7}  {'─'*6}  {'─'*65}")

        for rank, (score, doc) in enumerate(scored_docs):
            status  = "KEEP" if rank < top_n else "drop"
            preview = doc.page_content.replace("\n", " ")[:65]
            print(f"  {rank+1:<5} {score:>7.3f}  {status:<6}  {preview}")

    # ── Step 6: Return only top_n docs ───────────────────────────
    top_docs = [doc for _, doc in scored_docs[:top_n]]
    return top_docs


# ─────────────────────────────────────────────────────────────────
# BEFORE / AFTER COMPARISON UTILITY
# Shows clearly what changed after re-ranking.
# Great for your report and viva demo.
# ─────────────────────────────────────────────────────────────────

def compare_before_after(
    query      : str,
    before_docs: list,
    after_docs : list
) -> None:
    """
    Prints a side-by-side view of chunks before and after
    re-ranking. Use this in your demo to show the examiner
    exactly what the re-ranker removed and why that matters.
    """
    print(f"\n  Query: {query}")
    print(f"\n  BEFORE re-ranking ({len(before_docs)} chunks):")
    print(f"  {'─' * 55}")
    for i, doc in enumerate(before_docs):
        preview = doc.page_content.replace("\n", " ")[:70]
        print(f"  {i+1}. {preview}...")

    print(f"\n  AFTER re-ranking ({len(after_docs)} chunks):")
    print(f"  {'─' * 55}")
    for i, doc in enumerate(after_docs):
        preview = doc.page_content.replace("\n", " ")[:70]
        print(f"  {i+1}. {preview}...")

    removed = len(before_docs) - len(after_docs)
    print(f"\n  Removed {removed} low-relevance chunks "
          f"before sending to LLM")


# ─────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from loader              import load_documents
    from chunking            import chunk_documents
    from embeddings          import load_embedding_model
    from vector_store        import get_vector_store
    from adaptive_retrieval  import (
        build_bm25_index, adaptive_retrieve
    )

    print("=" * 60)
    print("RE-RANKER TEST")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────────
    print("\nSetting up pipeline...")
    docs          = load_documents()
    chunks        = chunk_documents(docs)
    model         = load_embedding_model()
    vs            = get_vector_store(chunks, model)
    bm25, chunks  = build_bm25_index(chunks)

    # ── Load re-ranker ─────────────────────────────────────────────
    print("\nLoading re-ranker...")
    reranker = load_reranker()

    # ── Test 1: Factual query ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 1 — Factual query")
    print("=" * 60)

    q1 = "What is tokenization?"

    retrieved_1, clf_1 = adaptive_retrieve(
        q1, vs, bm25, chunks, verbose=True
    )
    reranked_1 = rerank(
        query      = q1,
        docs       = retrieved_1,
        reranker   = reranker,
        query_type = clf_1["type"],
        verbose    = True
    )
    compare_before_after(q1, retrieved_1, reranked_1)

    # ── Test 2: Analytical query ───────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 2 — Analytical query")
    print("=" * 60)

    q2 = "Compare BM25 and FAISS retrieval methods."

    retrieved_2, clf_2 = adaptive_retrieve(
        q2, vs, bm25, chunks, verbose=True
    )
    reranked_2 = rerank(
        query      = q2,
        docs       = retrieved_2,
        reranker   = reranker,
        query_type = clf_2["type"],
        verbose    = True
    )
    compare_before_after(q2, retrieved_2, reranked_2)

    # ── Test 3: Multi-hop query ────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST 3 — Multi-hop query")
    print("=" * 60)

    q3 = ("How did early NLP methods lead to the "
          "development of transformer models?")

    retrieved_3, clf_3 = adaptive_retrieve(
        q3, vs, bm25, chunks, verbose=True
    )
    reranked_3 = rerank(
        query      = q3,
        docs       = retrieved_3,
        reranker   = reranker,
        query_type = clf_3["type"],
        verbose    = True
    )
    compare_before_after(q3, retrieved_3, reranked_3)

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Test 1 (factual)   : "
          f"{len(retrieved_1)} → {len(reranked_1)} chunks")
    print(f"  Test 2 (analytical): "
          f"{len(retrieved_2)} → {len(reranked_2)} chunks")
    print(f"  Test 3 (multi-hop) : "
          f"{len(retrieved_3)} → {len(reranked_3)} chunks")
    print(f"\n  ✓ Re-ranker working correctly")