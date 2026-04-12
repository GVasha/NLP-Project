import re
from typing import Set

from rag.config import INDEX_DIR, EMBEDDING_MODEL, TOP_K_FINAL, TOP_K_INITIAL
from rag.indexing.vectorstore import load_faiss_index


def extract_form_id(query: str) -> str | None:
    m = re.search(r"\bEX\s?(\d{2})\b", query, flags=re.IGNORECASE)
    return f"EX{m.group(1)}" if m else None


def extract_procedure_code(query: str) -> str | None:
    m = re.search(r"\b(\d{6})\b", query)
    return m.group(1) if m else None


def detect_intents(query: str) -> Set[str]:
    q = query.lower()
    intents = set()

    if any(x in q for x in ["document", "documents", "paperwork", "proof", "submit"]):
        intents.add("documentation")
    if any(x in q for x in ["where", "apply", "office", "station", "submit"]):
        intents.add("where_to_apply")
    if any(x in q for x in ["fee", "tax", "payment", "790", "012", "cost"]):
        intents.add("fees")
    if any(x in q for x in ["who can", "eligible", "qualify"]):
        intents.add("eligibility")
    if any(x in q for x in ["form", "ex13", "ex15", "ex16", "ex17", "ex18", "ex19", "ex29"]):
        intents.add("forms")

    if any(x in q for x in ["procedure code", "code of the procedure", "what is the code"]):
        intents.add("procedure_metadata")

    if any(x in q for x in ["how do i", "how to", "how can i", "steps", "carry out"]):
        intents.add("procedure_steps")

    if any(x in q for x in ["how long", "validity", "duration", "renewable"]):
        intents.add("duration_or_validity")

    return intents


def score_result(query: str, doc, rank_index: int = 0) -> int:
    score = 0
    q = query.lower()
    section_type = doc.metadata.get("section_type", "")
    title = (doc.metadata.get("title") or "").lower()
    form_ids = doc.metadata.get("form_ids", [])
    procedure_code = str(doc.metadata.get("procedure_code") or "")
    page_text = (doc.page_content or "").upper()

    intents = detect_intents(query)
    explicit_form = extract_form_id(query)
    explicit_proc = extract_procedure_code(query)

    # Preserve semantic ordering signal from the vector search.
    score += max(0, 6 - rank_index)

    if explicit_proc and explicit_proc == procedure_code:
        score += 12
    elif explicit_proc:
        score -= 4

    if explicit_form and explicit_form in page_text:
        score += 10
    elif explicit_form and explicit_form in form_ids:
        score += 4
    elif explicit_form:
        score -= 3

    if "documentation" in intents and section_type == "documentation":
        score += 5
    if "where_to_apply" in intents and section_type == "where_to_apply":
        score += 8
    if "fees" in intents and section_type == "fees":
        score += 5
    if "forms" in intents and section_type == "forms":
        score += 5
    if "eligibility" in intents and section_type in {"stakeholders", "requirements"}:
        score += 5
    if "procedure_metadata" in intents and section_type == "procedure_metadata":
        score += 9
    if "procedure_steps" in intents and section_type == "procedure_steps":
        score += 9
    if "procedure_steps" in intents and "nie" in q and section_type == "procedure_steps":
        score += 4
    if "duration_or_validity" in intents and section_type == "duration_or_validity":
        score += 6

    if "nie" in q and "nie" in title:
        score += 4
    if "student card" in q and "student card" in title:
        score += 4
    if "short-term stay" in q and "short-term stay" in title:
        score += 4

    if "where_to_apply" in intents and section_type in {"requirements", "documentation"}:
        score -= 4
    if "procedure_metadata" in intents and section_type in {"information", "general"}:
        score -= 3
    if "procedure_steps" in intents and section_type in {"requirements", "information"}:
        score -= 2
    if "procedure_steps" in intents and section_type == "requirements":
        score -= 3

    if section_type in {"appeals", "regulations", "classification", "competent_body", "process"}:
        score -= 1
    if section_type == "procedure_metadata":
        score -= 1

    return score


class SmartRetriever:
    def __init__(self):
        self.vectorstore = load_faiss_index(INDEX_DIR, EMBEDDING_MODEL)

    def search(self, query: str, k_initial: int = TOP_K_INITIAL, k_final: int = TOP_K_FINAL):
        effective_k_initial = max(k_initial, k_final * 12)
        initial_results = self.vectorstore.similarity_search(query, k=effective_k_initial)
        scored = [
            (score_result(query, doc, rank_index=i), doc)
            for i, doc in enumerate(initial_results)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k_final]]