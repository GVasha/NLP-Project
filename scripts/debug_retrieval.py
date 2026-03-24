from rag.retrieval.smart_retriever import SmartRetriever


def preview(text, n=400):
    text = text.replace("\n", " ").strip()
    return text[:n] + ("..." if len(text) > n else "")


def main():
    retriever = SmartRetriever()

    while True:
        question = input("\nQuery (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        docs = retriever.search(question)

        print(f"\nRetrieved {len(docs)} results:\n")

        for i, doc in enumerate(docs, start=1):
            meta = doc.metadata
            print(f"[{i}] {meta.get('title', 'Untitled')}")
            print(f"    Section: {meta.get('section_title', 'Unknown')}")
            print(f"    Procedure code: {meta.get('procedure_code', 'Unknown')}")
            print(f"    URL: {meta.get('source_url', 'Unknown')}")
            print(f"    Preview: {preview(doc.page_content)}")
            print()


if __name__ == "__main__":
    main()

# How do I obtain my student card?
# What documents do I need?
# Where can i process my application?
# How long does it take to get my card?
# I haven't started my application yet, but the deadline is approaching. What should I do?
# I submitted my application but haven't received any updates. What should I do?