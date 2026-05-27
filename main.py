import os
import sys
import time

from classifier import classify_query
from config import DATA_DIR
from rag_pipeline import answer_query, prepare_vector_store


def setup_pipeline(force_rebuild: bool = False) -> dict:
    print("\n" + "=" * 60)
    print("  QUERY-AWARE RAG SYSTEM")
    print("  Setting up pipeline...")
    print("=" * 60)

    total_start = time.time()

    if force_rebuild and os.path.isdir("faiss_index"):
        for filename in ("index.faiss", "metadata.pkl"):
            path = os.path.join("faiss_index", filename)
            if os.path.exists(path):
                os.remove(path)

    store, build_info = prepare_vector_store(data_mode="existing")
    elapsed = round(time.time() - total_start, 1)

    print(f"\n  Index status : {build_info.get('index_status', 'unknown')}")
    print(f"  Data source  : {build_info.get('data_label', DATA_DIR)}")
    print(f"  Documents    : {build_info.get('document_count', 0)}")
    print(f"  Chunks       : {build_info.get('chunk_count', 0)}")
    print(f"  Ready in     : {elapsed}s\n")

    return {
        "store": store,
        "build_info": build_info,
    }


def run_query(query: str, pipeline: dict, verbose: bool = True) -> dict:
    clf_result = classify_query(query)
    result = answer_query(
        query=query,
        store=pipeline["store"],
        top_k=clf_result["top_k"],
        use_expansion=True,
        api_key=os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY"),
    )

    retrieval = result["retrieval"]
    generation = result["generation"]

    if verbose:
        print(f"\n  Classified as : {clf_result['type'].upper()}")
        print(f"  Strategy      : {clf_result['strategy']}")
        print(f"  Reason        : {clf_result['reason']}")
        print(f"  Retrieved     : {retrieval['top_k']} chunks")
        print(f"  Confidence    : {retrieval['confidence']:.3f}")
        print(f"  Latency       : {result['total_latency_ms']:.1f} ms")

    return {
        "query": query,
        "query_type": clf_result["type"],
        "strategy": clf_result["strategy"],
        "docs_retrieved": retrieval["top_k"],
        "answer": generation["answer"],
        "sources": result["sources"],
        "scores": retrieval["scores"],
        "chunks": retrieval["chunks"],
        "retrieval_ms": round(retrieval["latency_ms"], 1),
        "generate_ms": round(generation["latency_ms"], 1),
        "total_ms": round(result["total_latency_ms"], 1),
        "evaluation": generation["evaluation"],
        "performance_metrics": generation.get("performance_metrics", {}),
        "supporting_evidence": generation.get("supporting_evidence", []),
    }


def print_answer(result: dict) -> None:
    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result["answer"])

    if result["sources"]:
        print("\nSources:")
        for source in dict.fromkeys(result["sources"]):
            print(f"  - {source}")

    if result["scores"]:
        print("\nSimilarity scores:")
        for idx, score in enumerate(result["scores"], start=1):
            print(f"  {idx}. {score:.3f}")

    evaluation = result.get("evaluation", {})
    if evaluation:
        print("\nEvaluation:")
        print(f"  Verdict    : {evaluation.get('verdict', 'UNKNOWN')}")
        print(f"  Confidence : {evaluation.get('confidence', 0.0):.2f}")
        print(f"  Explanation: {evaluation.get('explanation', '')}")

    performance = result.get("performance_metrics", {})
    if performance:
        print("\nPerformance Metrics:")
        print(f"  Retrieval confidence : {performance.get('retrieval_confidence', 0.0):.2f}")
        print(f"  Groundedness         : {performance.get('groundedness_score', 0.0):.2f}")
        print(f"  Answer accuracy      : {performance.get('answer_accuracy', 0.0):.2f}")
        print(f"  Citation coverage    : {performance.get('citation_coverage', 0.0):.2f}")
        print(f"  Hallucination risk   : {performance.get('hallucination_risk', 0.0):.2f}")

    evidence = result.get("supporting_evidence", [])
    if evidence:
        print("\nProof From Retrieved Sources:")
        for item in evidence:
            print(
                f"  - Source {item['rank']} ({item['source']}) | "
                f"support {item['support_score']:.2f} | similarity {item['similarity']:.2f}"
            )
            print(f"    Cited in answer: {item['cited_in_answer']}")
            print(f"    Proof: {item['proof']}")

    print(f"\nTotal latency: {result['total_ms']:.1f} ms")


def interactive_loop(pipeline: dict) -> None:
    print("=" * 60)
    print("  INTERACTIVE MODE")
    print("  Type a question and press Enter.")
    print("  Commands: 'quit' | 'help'")
    print("=" * 60)

    while True:
        try:
            print()
            query = input("  Your question: ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                print("\n  Goodbye!\n")
                break

            if query.lower() == "help":
                print("\n  Commands:")
                print("    quit -> exit")
                print("    help -> show commands")
                continue

            result = run_query(query, pipeline, verbose=True)
            print_answer(result)

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Type 'quit' to exit.")
        except Exception as exc:
            print(f"\n  Error: {exc}")


if __name__ == "__main__":
    force_rebuild = "--build" in sys.argv
    pipeline = setup_pipeline(force_rebuild=force_rebuild)
    interactive_loop(pipeline)
