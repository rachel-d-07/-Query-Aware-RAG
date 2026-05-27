"""
generator.py — Answer generation + LLM self-evaluation using Google Gemini (free tier)
"""

import os
import re
import time
from typing import List, Dict, Any, Optional

from config import LLM_MODEL
from env_utils import load_local_env

# ── LLM backend ───────────────────────────────────────────────────────────────
# Uses Google Gemini (free at ai.google.dev).
# Swap the _call_llm() function to use any other backend (OpenAI, HuggingFace, etc.)

load_local_env()

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


DEFAULT_GEMINI_MODELS = [
    os.getenv("GEMINI_MODEL", "").strip(),
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "").strip() or LLM_MODEL
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower()


def configure_gemini(api_key: str):
    """Call once at app startup with your Gemini API key."""
    if not GENAI_AVAILABLE:
        raise ImportError("Install google-generativeai:  pip install google-generativeai")
    genai.configure(api_key=api_key)


# ── Core generation ───────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: List[dict],
    scores: List[float],
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate an answer from retrieved chunks and evaluate its correctness.

    Returns:
        answer          — LLM-generated answer string
        evaluation      — dict with 'verdict', 'confidence', 'explanation'
        latency_ms      — generation time in milliseconds
        context_used    — number of chunks used
    """
    context = _build_context(chunks, scores)

    # Step 1 — Generate answer
    t0 = time.time()
    answer = _generate(query, context, api_key=api_key)
    gen_latency = (time.time() - t0) * 1000

    # Step 2 — Self-evaluate
    evaluation = _evaluate(query, answer, context, api_key=api_key)
    source_comparison = _build_source_comparison(answer, chunks, scores)
    performance_metrics = _build_performance_metrics(answer, scores, evaluation, source_comparison)
    supporting_evidence = _build_supporting_evidence(source_comparison)

    return {
        "answer":       answer,
        "evaluation":   evaluation,
        "latency_ms":   gen_latency,
        "context_used": len(chunks),
        "context":      context,
        "source_comparison": source_comparison,
        "performance_metrics": performance_metrics,
        "supporting_evidence": supporting_evidence,
    }


# ── Prompts ───────────────────────────────────────────────────────────────────

def _build_context(chunks: List[dict], scores: List[float]) -> str:
    parts = []
    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        parts.append(
            f"[Source {i} | File: {chunk.get('source','?')} | Score: {score:.3f}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _generate(query: str, context: str, api_key: Optional[str] = None) -> str:
    prompt = f"""You are a helpful assistant answering questions from retrieved documents.

CONTEXT (retrieved document chunks):
{context}

QUESTION: {query}

Instructions:
- Answer based ONLY on the context above.
- If the answer is not in the context, say "I could not find an answer in the provided documents."
- Be concise and factual.
- Every important claim must include an inline citation using the source labels like [Source 1].
- Do not cite any source that is not present in the context.

ANSWER:"""

    return _call_llm(prompt, api_key=api_key)


def _evaluate(query: str, answer: str, context: str, api_key: Optional[str] = None) -> dict:
    """Ask the LLM to judge whether its own answer is supported by the context."""
    prompt = f"""You are an evaluator checking whether an AI answer is supported by the retrieved context.

QUESTION: {query}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Evaluate the answer and respond in this exact format:
VERDICT: [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
CONFIDENCE: [a number from 0.0 to 1.0]
EXPLANATION: [one sentence explaining your verdict]"""

    raw = _call_llm(prompt, api_key=api_key)
    return _parse_evaluation(raw)


def _parse_evaluation(raw: str) -> dict:
    result = {"verdict": "UNKNOWN", "confidence": 0.0, "explanation": raw}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            result["verdict"] = line.replace("VERDICT:", "").strip()
        elif line.startswith("CONFIDENCE:"):
            try:
                result["confidence"] = float(line.replace("CONFIDENCE:", "").strip())
            except ValueError:
                pass
        elif line.startswith("EXPLANATION:"):
            result["explanation"] = line.replace("EXPLANATION:", "").strip()
    return result


def _build_source_comparison(answer: str, chunks: List[dict], scores: List[float]) -> List[dict]:
    cited_sources = set(_extract_cited_sources(answer))
    answer_terms = {
        token.strip(".,:;!?()[]{}\"'").lower()
        for token in answer.split()
        if len(token.strip(".,:;!?()[]{}\"'")) > 3
    }

    comparison = []
    for index, (chunk, score) in enumerate(zip(chunks, scores), start=1):
        chunk_terms = {
            token.strip(".,:;!?()[]{}\"'").lower()
            for token in chunk["text"].split()
            if len(token.strip(".,:;!?()[]{}\"'")) > 3
        }
        overlap = answer_terms.intersection(chunk_terms)
        overlap_ratio = round(len(overlap) / max(len(answer_terms), 1), 3)
        evidence_terms = sorted(overlap)[:12]
        citation_used = index in cited_sources
        support_score = round((0.6 * float(score)) + (0.4 * overlap_ratio), 3)

        comparison.append({
            "rank": index,
            "source": chunk.get("source", "?"),
            "similarity": round(float(score), 3),
            "answer_overlap": overlap_ratio,
            "support_score": support_score,
            "cited_in_answer": citation_used,
            "matching_terms": evidence_terms,
            "excerpt": chunk["text"][:240],
        })

    return comparison


def _build_performance_metrics(
    answer: str,
    scores: List[float],
    evaluation: Dict[str, Any],
    source_comparison: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cited_sources = set(_extract_cited_sources(answer))
    citation_coverage = (
        sum(1 for item in source_comparison if item["cited_in_answer"]) / max(len(source_comparison), 1)
    )
    overlap_score = (
        sum(item["answer_overlap"] for item in source_comparison) / max(len(source_comparison), 1)
    )
    retrieval_strength = (
        sum(scores[:3]) / min(len(scores), 3)
        if scores else 0.0
    )

    evaluation_conf = float(evaluation.get("confidence", 0.0) or 0.0)
    verdict = str(evaluation.get("verdict", "UNKNOWN")).upper()
    verdict_weight = {
        "SUPPORTED": 1.0,
        "PARTIALLY SUPPORTED": 0.65,
        "NOT SUPPORTED": 0.2,
        "UNKNOWN": 0.35,
    }.get(verdict, 0.35)

    groundedness = min(
        1.0,
        (0.35 * retrieval_strength) +
        (0.30 * overlap_score) +
        (0.20 * citation_coverage) +
        (0.15 * evaluation_conf)
    )
    answer_accuracy = min(1.0, (0.55 * verdict_weight) + (0.45 * groundedness))
    hallucination_risk = max(0.0, 1.0 - answer_accuracy)

    return {
        "retrieval_confidence": round(retrieval_strength, 3),
        "groundedness_score": round(groundedness, 3),
        "answer_accuracy": round(answer_accuracy, 3),
        "citation_coverage": round(citation_coverage, 3),
        "hallucination_risk": round(hallucination_risk, 3),
        "cited_source_count": len(cited_sources),
        "verdict": verdict,
    }


def _build_supporting_evidence(source_comparison: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        source_comparison,
        key=lambda item: (item["cited_in_answer"], item["support_score"], item["similarity"]),
        reverse=True,
    )
    evidence = []
    for item in ranked[:3]:
        evidence.append({
            "rank": item["rank"],
            "source": item["source"],
            "support_score": item["support_score"],
            "similarity": item["similarity"],
            "cited_in_answer": item["cited_in_answer"],
            "proof": item["excerpt"],
            "matching_terms": item["matching_terms"],
        })
    return evidence


def _extract_cited_sources(answer: str) -> List[int]:
    return [int(match) for match in re.findall(r"\[Source\s+(\d+)\]", answer, flags=re.IGNORECASE)]


# ── LLM call (swap here to change backend) ────────────────────────────────────

def _call_llm(prompt: str, api_key: Optional[str] = None) -> str:
    provider, resolved_api_key = _resolve_provider(api_key)
    if provider == "groq":
        return _call_groq(prompt, resolved_api_key)
    if provider == "gemini":
        return _call_gemini(prompt, resolved_api_key)
    return (
        "[LLM Error: no API key found. Add GROQ_API_KEY or GEMINI_API_KEY to your environment "
        "or .env file.]"
    )


def _resolve_provider(api_key: Optional[str]) -> tuple[str, str]:
    explicit_key = (api_key or "").strip()
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

    if explicit_key:
        if explicit_key.startswith("gsk_"):
            return "groq", explicit_key
        return "gemini", explicit_key

    if DEFAULT_PROVIDER == "groq" and groq_key:
        return "groq", groq_key
    if DEFAULT_PROVIDER == "gemini" and gemini_key:
        return "gemini", gemini_key

    if groq_key:
        return "groq", groq_key
    if gemini_key:
        return "gemini", gemini_key
    return "", ""


def _call_groq(prompt: str, api_key: str) -> str:
    if not GROQ_AVAILABLE:
        return "[Groq SDK not installed. Run: pip install groq]"
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=DEFAULT_GROQ_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        return f"[LLM Error using Groq model {DEFAULT_GROQ_MODEL}: {e}]"


def _call_gemini(prompt: str, api_key: str) -> str:
    if not GENAI_AVAILABLE:
        return "[google-generativeai not installed. Run: pip install google-generativeai]"

    configure_gemini(api_key)
    attempted_models = []
    for model_name in [name for name in DEFAULT_GEMINI_MODELS if name]:
        attempted_models.append(model_name)
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_text = str(e)
            if "404" in error_text or "not found" in error_text.lower():
                continue
            return f"[LLM Error using {model_name}: {e}]"

    return (
        "[LLM Error: no supported Gemini text model was available. "
        f"Tried: {', '.join(attempted_models)}. "
        "Set GEMINI_MODEL in your .env to a currently available model, for example gemini-2.5-flash.]"
    )
