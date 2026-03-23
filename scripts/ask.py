import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from rag.service.qa_service import QAService

def main():
    service = QAService()

    while True:
        question = input("\nAsk a question (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        result = service.ask(question)

        print("\nANSWER:\n")
        print(result["answer"])

        print("\nSOURCES:\n")
        for i, s in enumerate(result["sources"], start=1):
            print(f"[{i}] {s['title']} | {s['section']} | {s['procedure_code']}")
            print(f"    {s['source_url']}")


if __name__ == "__main__":
    main()