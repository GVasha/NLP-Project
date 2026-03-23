from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.indexing.vectorstore import build_faiss_index_from_chunks, save_faiss_index


CHUNKS_PATH = "storage/processed/chunks.jsonl"
INDEX_DIR = "storage/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main():
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

    print("Building FAISS index from chunks...")
    vectorstore, n_docs = build_faiss_index_from_chunks(
        chunks_path=CHUNKS_PATH,
        model_name=EMBEDDING_MODEL,
    )

    save_faiss_index(vectorstore, INDEX_DIR)

    print(f"Indexed {n_docs} chunks")
    print(f"Saved FAISS index to {INDEX_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")


if __name__ == "__main__":
    main()