import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.indexing.chunker import load_canonical_docs, chunk_documents


DOCS_DIR = "storage/processed/docs"
OUTPUT_PATH = Path("storage/processed/chunks.jsonl")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    documents = load_canonical_docs(DOCS_DIR)
    print(f"Loaded {len(documents)} canonical documents")

    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print(f"Saved {len(chunks)} chunks to {OUTPUT_PATH}")

    # tiny summary
    by_doc = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, 0)
        by_doc[chunk.doc_id] += 1

    for doc_id, count in sorted(by_doc.items()):
        print(f"{doc_id}: {count} chunks")


if __name__ == "__main__":
    main()