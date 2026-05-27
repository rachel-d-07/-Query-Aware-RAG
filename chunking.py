from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE


def clean_text(text: str) -> str:
    """Remove obvious low-value lines before chunking."""
    lines = text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 10:
            continue

        lower_line = line.lower()
        if lower_line.startswith(("q:", "ans:", "answer:", "option ")):
            continue
        if "import " in line or "def " in line:
            continue
        clean_lines.append(line)

    return "\n".join(clean_lines)


def chunk_documents(docs):
    """Split cleaned documents into retrieval-friendly chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = []

    for doc in docs:
        cleaned = clean_text(doc.page_content)
        if not cleaned.strip():
            continue
        splits = splitter.split_text(cleaned)

        for chunk in splits:
            if len(chunk.strip()) < 50:
                continue
            metadata = dict(doc.metadata or {})
            metadata["chunk_id"] = len(chunks)
            metadata["char_length"] = len(chunk)
            new_doc = Document(page_content=chunk, metadata=metadata)
            chunks.append(new_doc)

    print(f"  ✓ Created {len(chunks)} clean chunks")
    return chunks
