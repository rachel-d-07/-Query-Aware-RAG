"""
retriever.py — Query-aware retrieval with scoring and optional query expansion
"""

from typing import Dict, Any, List, Set

from vector_store import VectorStore


class Retriever:
    """
    Wraps VectorStore with query pre-processing and result packaging.
    """

    # Simple domain-aware keyword expansions (rule-based)
    EXPANSIONS = {
        "ai":        ["artificial intelligence", "machine learning"],
        "ml":        ["machine learning", "model training"],
        "nlp":       ["natural language processing", "text analysis"],
        "llm":       ["large language model", "generative AI"],
        "bias":      ["fairness", "discrimination", "skewed"],
        "rag":       ["retrieval augmented generation", "document retrieval"],
        "vector":    ["embedding", "similarity search"],
        "model":     ["algorithm", "neural network"],
        "data":      ["dataset", "training examples"],
        "accuracy":  ["precision", "recall", "performance"],
    }

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.store = vector_store
        self.top_k = top_k
        self.stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
            "was", "what", "when", "where", "which", "who", "why", "with",
        }

    # ── Query optimisation ────────────────────────────────────────────────────

    def expand_query(self, query: str) -> str:
        """Rule-based query expansion — adds related terms for short/vague queries."""
        words = query.lower().split()
        extras = []
        for word in words:
            key = word.strip(".,?!")
            if key in self.EXPANSIONS:
                extras.extend(self.EXPANSIONS[key])
        if extras:
            return query + " " + " ".join(extras)
        return query

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        use_expansion: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve top-k chunks for a query.

        Returns a result dict with:
            original_query, expanded_query, chunks, scores,
            confidence, latency_ms, top_k
        """
        if not self.store.is_ready():
            raise ValueError("Vector store is not ready. Index is empty.")

        original_query = query
        expanded_query = self.expand_query(query) if use_expansion else query

        fetch_k = min(max(self.top_k * 3, self.top_k), self.store.total_chunks)
        chunks, scores, latency_ms = self.store.retrieve(expanded_query, top_k=fetch_k)
        reranked = self._rerank_chunks(original_query, chunks, scores)[: self.top_k]

        if reranked:
            chunks = [item["chunk"] for item in reranked]
            scores = [item["score"] for item in reranked]
        else:
            chunks = chunks[: self.top_k]
            scores = scores[: self.top_k]

        # Confidence = mean of top scores (or top-3 if k > 3)
        top_scores = scores[:3] if len(scores) >= 3 else scores
        confidence = float(sum(top_scores) / len(top_scores)) if top_scores else 0.0

        return {
            "original_query":  original_query,
            "expanded_query":  expanded_query,
            "query_expanded":  expanded_query != original_query,
            "chunks":          chunks,
            "scores":          scores,
            "confidence":      confidence,
            "latency_ms":      latency_ms,
            "top_k":           len(chunks),
        }

    def _rerank_chunks(
        self,
        query: str,
        chunks: List[dict],
        scores: List[float],
    ) -> List[Dict[str, Any]]:
        query_terms = self._tokenise(query)
        lowered_query = query.lower()
        reranked = []

        for chunk, semantic_score in zip(chunks, scores):
            chunk_text = chunk.get("text", "")
            chunk_terms = self._tokenise(chunk_text)

            overlap = len(query_terms & chunk_terms)
            coverage = overlap / max(len(query_terms), 1)
            phrase_bonus = 0.12 if lowered_query and lowered_query in chunk_text.lower() else 0.0
            lexical_score = min(1.0, coverage + phrase_bonus)
            combined_score = (0.8 * float(semantic_score)) + (0.2 * lexical_score)

            reranked.append({
                "chunk": chunk,
                "score": round(combined_score, 4),
            })

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked

    def _tokenise(self, text: str) -> Set[str]:
        tokens = {
            word.strip(".,?!:;()[]{}\"'").lower()
            for word in text.split()
        }
        return {
            token for token in tokens
            if len(token) > 2 and token not in self.stopwords
        }
