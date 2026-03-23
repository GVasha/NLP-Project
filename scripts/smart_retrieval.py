import re
from typing import List, Tuple

from rag.indexing.vectorstore import load_faiss_index


INDEX_DIR = "storage/faiss_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def extract_form_id(query: str) -> str | None:
    m = re.search(r"\bEX\s?(\d{2})\b", query, flags=re.IGNORECASE)
    if m:
        return f"EX{m.group(1)}"
    return None


def detect_intents(query: str) -> set[str]:
    q = query.lower()
    intents = set()

    if any(x in q for x in ["document", "documents", "paperwork", "proof", "bring", "need to submit"]):
        intents.add("documentation")
    if any(x in q for x in ["where", "apply", "submit", "office", "station", "process my application"]):
        intents.add("where_to_apply")
    if any(x in q for x in ["fee", "tax", "payment", "790", "012", "cost"]):
        intents.add("fees")
    if any(x in q for x in ["who can", "eligible", "stakeholder", "request", "qualify"]):
        intents.add("eligibility")
    if any(x in q for x in ["form", "ex15", "ex16", "ex17", "ex18", "ex19", "ex29", "ex13"]):
        intents.add("forms")

    return intents


def score_result(query: str, doc) -> int:
    score = 0
    q = query.lower()
    section_type = doc.metadata.get("section_type", "")
    title = (doc.metadata.get("title") or "").lower()
    form_ids = doc.metadata.get("form_ids", [])

    intents = detect_intents(query)
    explicit_form = extract_form_id(query)

    if explicit_form and explicit_form in form_ids:
        score += 10
    elif explicit_form:
        score -= 3

    if "documentation" in intents and section_type == "documentation":
        score += 5
    if "where_to_apply" in intents and section_type == "where_to_apply":
        score += 5
    if "fees" in intents and section_type == "fees":
        score += 5
    if "forms" in intents and section_type == "forms":
        score += 5
    if "eligibility" in intents and section_type in {"stakeholders", "requirements"}:
        score += 5

    if "nie" in q and "nie" in title:
        score += 4
    if "student card" in q and "student card" in title:
        score += 4
    if "short-term stay" in q and "short-term stay" in title:
        score += 4

    # small penalties for weak sections unless query matches them
    if section_type in {"appeals", "regulations"} and not intents.intersection({"fees"}):
        score -= 2
    if section_type == "procedure_metadata":
        score -= 1

    return score


def smart_search(vectorstore, query: str, k_initial: int = 10, k_final: int = 3):
    initial_results = vectorstore.similarity_search(query, k=k_initial)

    scored: List[Tuple[int, object]] = []
    for doc in initial_results:
        score = score_result(query, doc)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:k_final]]


def main():
    vectorstore = load_faiss_index(INDEX_DIR, EMBEDDING_MODEL)

    test_queries = [
        "What documents do I need to extend a short-term stay?",
        "Where can I apply for a student card for foreigners?",
        "What is form EX17 for?",
        "What is fee 790 code 012?",
        "Who can request an NIE?",
    ]

    for query in test_queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}\n")

        results = smart_search(vectorstore, query)

        for i, doc in enumerate(results, start=1):
            print(f"[Result {i}]")
            print(f"title: {doc.metadata.get('title')}")
            print(f"section: {doc.metadata.get('section_title')}")
            print(f"section_type: {doc.metadata.get('section_type')}")
            print(f"procedure_code: {doc.metadata.get('procedure_code')}")
            print(f"form_ids: {doc.metadata.get('form_ids')}")
            print("\nTEXT:")
            print(doc.page_content[:1000])
            print("\n" + "-" * 100)


if __name__ == "__main__":
    main()