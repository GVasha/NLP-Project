import sys
from anyio import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.indexing.vectorstore import load_faiss_index

INDEX_DIR = "storage/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


TEST_QUERIES = [
    "What documents do I need to extend a short-term stay?",
    "Where can I apply for a student card for foreigners?",
    "What is form EX17 for?",
    "What is fee 790 code 012?",
    "Who can request an NIE?",
]


def main():
    vectorstore = load_faiss_index(INDEX_DIR, EMBEDDING_MODEL)

    for query in TEST_QUERIES:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}\n")

        results = vectorstore.similarity_search(query, k=3)

        for i, doc in enumerate(results, start=1):
            print(f"[Result {i}]")
            print(f"title: {doc.metadata.get('title')}")
            print(f"section: {doc.metadata.get('section_title')}")
            print(f"section_type: {doc.metadata.get('section_type')}")
            print(f"procedure_code: {doc.metadata.get('procedure_code')}")
            print(f"form_ids: {doc.metadata.get('form_ids')}")
            print(f"source_url: {doc.metadata.get('source_url')}")
            print("\nTEXT:")
            print(doc.page_content[:1200])
            print("\n" + "-" * 100)


if __name__ == "__main__":
    main()