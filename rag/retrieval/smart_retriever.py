import re
from typing import List

from rag.config import INDEX_DIR, EMBEDDING_MODEL, TOP_K_FINAL, TOP_K_INITIAL
from rag.indexing.vectorstore import load_faiss_index


def extract_form_id(query: str) -> str | None:
    m = re.search(r"\bEX\s?(\d{2})\b", query, flags=re.IGNORECASE)
    return f"EX{m.group(1)}" if m else None


def detect_intents(query: str) -> set[str]:
    q = query.lower()
    intents = set()

    if any(x in q for x in ["document", "documents", "paperwork", "proof", "submit"]):
        intents.add("documentation")
    if any(x in q for x in ["where", "apply", "office", "station", "submit"]):
        intents.add("where_to_apply")
    if any(x in q for x in ["fee", "tax", "payment", "790", "012", "cost"]):
        intents.add("fees")
    if any(x in q for x in ["who can", "eligible", "request", "qualify"]):
        intents.add("eligibility")
    if any(x in q for x in ["form", "ex13", "ex15", "ex16", "ex17", "ex18", "ex19", "ex29"]):
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

    if section_type in {"appeals", "regulations", "classification", "competent_body", "process"}:
        score -= 1
    if section_type == "procedure_metadata":
        score -= 1

    return score


class SmartRetriever:
    def __init__(self):
        self.vectorstore = load_faiss_index(INDEX_DIR, EMBEDDING_MODEL)

    def search(self, query: str, k_initial: int = TOP_K_INITIAL, k_final: int = TOP_K_FINAL):
        initial_results = self.vectorstore.similarity_search(query, k=k_initial)
        scored = [(score_result(query, doc), doc) for doc in initial_results]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k_final]]