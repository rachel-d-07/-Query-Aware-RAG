# embeddings.py
# Responsibility: Load the embedding model and convert
#                 text into numeric vectors.
# Input        : A string or list of strings
# Output       : A LangChain-compatible embedding model object
#                (used by vector_store.py to build the FAISS index)

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


def load_embedding_model() -> HuggingFaceEmbeddings:
    """
    Loads the sentence-transformer embedding model.

    First call  : downloads ~80MB from HuggingFace, then caches it.
    Later calls : loads instantly from local cache.

    Returns:
        HuggingFaceEmbeddings object — compatible with FAISS
        and all LangChain vector stores.
    """
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    print(f"  (First run downloads ~80MB — subsequent runs are instant)")

    embedding_model = HuggingFaceEmbeddings(
        model_name  = EMBEDDING_MODEL,
        model_kwargs = {"device": "cpu"},   # use "cuda" if you have a GPU
        encode_kwargs = {"normalize_embeddings": True},
        # normalize_embeddings=True makes cosine similarity
        # equivalent to dot product — slightly faster search
    )

    print(f"  ✓ Embedding model loaded")
    return embedding_model


def embed_text(text: str, embedding_model: HuggingFaceEmbeddings) -> list:
    """
    Converts a single string into a vector.
    Useful for testing and for embedding queries at search time.

    Args:
        text            : any string
        embedding_model : loaded model from load_embedding_model()

    Returns:
        list of floats (length 384 for MiniLM-L6-v2)
    """
    vector = embedding_model.embed_query(text)
    return vector


def embed_batch(texts: list, embedding_model: HuggingFaceEmbeddings) -> list:
    """
    Converts a list of strings into a list of vectors.
    More efficient than calling embed_text() in a loop.

    Args:
        texts           : list of strings
        embedding_model : loaded model from load_embedding_model()

    Returns:
        list of vectors (each vector is a list of 384 floats)
    """
    vectors = embedding_model.embed_documents(texts)
    return vectors


# ── Quick self-test ───────────────────────────────────────────────
# Run this file directly to verify it works:
# $ python embeddings.py

if __name__ == "__main__":
    print("=" * 50)
    print("EMBEDDINGS TEST")
    print("=" * 50)

    # Step 1: Load model
    print("\nLoading model...")
    model = load_embedding_model()

    # Step 2: Embed a single query
    print("\nEmbedding a single sentence...")
    vec = embed_text("What is natural language processing?", model)
    print(f"  Vector dimensions : {len(vec)}")
    print(f"  First 5 values    : {[round(v, 4) for v in vec[:5]]}")

    # Step 3: Prove similarity works
    print("\nSimilarity test:")
    print("  (Similar sentences should have high score,")
    print("   different sentences should have low score)\n")

    import numpy as np

    sentences = [
        "What is natural language processing?",   # query
        "NLP is a field of artificial intelligence.",  # similar
        "Deep learning models process text data.",     # somewhat related
        "The Eiffel Tower is in Paris.",               # unrelated
    ]

    query_vec = np.array(embed_text(sentences[0], model))

    for sentence in sentences[1:]:
        other_vec = np.array(embed_text(sentence, model))
        # Cosine similarity: 1.0 = identical, 0.0 = unrelated
        similarity = float(np.dot(query_vec, other_vec))
        bar = "█" * int(similarity * 30)
        print(f"  {similarity:.3f} {bar}")
        print(f"         \"{sentence}\"")