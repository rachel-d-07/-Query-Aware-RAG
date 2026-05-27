import re

from config import RETRIEVAL_CONFIG


MULTI_HOP_KEYWORDS = [
    "how did", "how does",
    "why did", "why does",
    "relationship between",
    "connection between",
    "impact of",
    "effect of",
    "led to", "resulted in",
    "evolution of",
    "history of",
    "role of",
    "influence of",
]


MULTI_HOP_TRIGGERS = [
    r"\b(lead to|leads to|led to|result in|resulted in|cause|caused|contribute to|contributed to)\b",
    r"\b(relation|relationship|connection|link|association)\b.{0,30}\bbetween\b",
    r"\bwho (invented|discovered|developed).{5,60}\b(then|later|after)\b",
    r"\b(and (what|who|how|why|when)|as well as)\b",
]


ANALYTICAL_TRIGGERS = [
    "compare", "contrast", "difference between",
    "similarities between", "versus", " vs ",
    "explain",
    "analyze", "analyse", "evaluate", "assess",
    "discuss", "examine", "elaborate",
    "advantages", "disadvantages",
    "pros and cons", "benefits",
    "summarize", "overview",
    "types of", "steps of",
]


FACTUAL_SIGNALS = [
    "what is", "what are", "who is", "who are",
    "when was", "when did", "where is",
    "define", "meaning of", "full form of",
]


def classify_query(query: str) -> dict:
    """Classify a query into factual, analytical, or multi-hop."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    q = query.lower().strip()
    word_count = len(q.split())

    for keyword in MULTI_HOP_KEYWORDS:
        if keyword in q:
            return _build_result(
                qtype="multi_hop",
                reason=f"Matched multi-hop keyword: '{keyword}'",
            )

    for pattern in MULTI_HOP_TRIGGERS:
        if re.search(pattern, q):
            return _build_result(
                qtype="multi_hop",
                reason="Matched multi-hop pattern",
            )

    if word_count > 18 and _has_multiple_questions(q):
        return _build_result(
            qtype="multi_hop",
            reason="Long query with multiple parts",
        )

    for keyword in ANALYTICAL_TRIGGERS:
        if keyword in q:
            return _build_result(
                qtype="analytical",
                reason=f"Matched analytical keyword: '{keyword}'",
            )

    if word_count > 12:
        return _build_result(
            qtype="analytical",
            reason=f"Long query ({word_count} words)",
        )

    return _build_result(
        qtype="factual",
        reason="Short direct query",
    )


def _has_multiple_questions(q: str) -> bool:
    """Return True when a query appears to combine multiple asks."""
    connectors = [
        " and ",
        " also ",
        " as well as ",
        " additionally ",
        " furthermore ",
        " plus ",
    ]
    return any(c in q for c in connectors)


def _build_result(qtype: str, reason: str) -> dict:
    config = RETRIEVAL_CONFIG[qtype]
    return {
        "type": qtype,
        "reason": reason,
        "top_k": config["top_k"],
        "strategy": config["strategy"],
        "config": config,
    }


def print_classification(query: str, result: dict) -> None:
    """Print a readable summary for CLI/debug use."""
    print(f"\nQuery   : {query}")
    print(f"Type    : {result['type']}")
    print(f"Top-k   : {result['top_k']}")
    print(f"Strategy: {result['strategy']}")
    print(f"Reason  : {result['reason']}")
