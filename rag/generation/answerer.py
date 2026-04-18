import re

from ollama import chat

from rag.config import OLLAMA_MODEL


SYSTEM_PROMPT = """You answer immigration/admin procedure questions using only the provided context.
Rules:
- Use only the retrieved context.
- If the answer is not clearly supported in the context, explicitly say there is not enough information in the retrieved sources.
- Do not invent details, URLs, timings, contact data, prices, legal guarantees, or policies not stated in context.
- Keep answers concise and practical.
- Prefer bullet points for steps/documents.
- End with a short Sources section listing title + section + procedure code.
"""


UNCERTAINTY_RESPONSE = (
    "There is not enough information in the retrieved sources to answer this safely. "
    "Please rephrase your question with specific procedure details, form code, or document type."
)

QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "that", "the", "to", "with", "this", "these", "those", "you", "your",
    "can", "i", "we", "what", "where", "how", "who", "when", "which",
}


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9-]+", normalize(text))
    return [t for t in tokens if len(t) > 2 and t not in QUERY_STOPWORDS]


def detect_ambiguous_or_oos_query(question: str) -> bool:
    q = normalize(question)
    if any(x in q for x in [
        "weather", "rent", "exchange rate", "whatsapp", "book my appointment",
        "guarantee", "guaranteed", "internal confidence", "faster city", "private email",
        "submit my application for me", "court strategy", "same-day premium",
    ]):
        return True

    # Very short generic questions are often under-specified and should be handled conservatively.
    if len(tokenize(q)) <= 3 and any(x in q for x in ["documents", "apply", "where", "what do i need"]):
        return True

    return False


def retrieval_overlap_ratio(question: str, docs) -> float:
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return 0.0

    retrieval_tokens = set()
    for d in docs:
        retrieval_tokens.update(tokenize(d.page_content))

    if not retrieval_tokens:
        return 0.0

    return len(q_tokens & retrieval_tokens) / len(q_tokens)


def detect_uncertainty(answer: str) -> bool:
    patterns = [
        r"\bnot enough (information|context|evidence)\b",
        r"\binsufficient (information|context|evidence)\b",
        r"\bcannot (determine|confirm|answer|find)\b",
        r"\bcan\'?t (determine|confirm|answer|find)\b",
        r"\bnot available in the retrieved sources\b",
        r"\bdo not know\b",
        r"\bdon\'?t know\b",
    ]
    text = normalize(answer)
    return any(re.search(p, text) for p in patterns)


def grounded_sentence_ratio(answer: str, docs, min_overlap: int = 2) -> tuple[float, int, int]:
    retrieval_tokens = set()
    for d in docs:
        retrieval_tokens.update(tokenize(d.page_content))

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    claims = 0
    supported = 0

    for s in sentences:
        if detect_uncertainty(s):
            continue
        st = set(tokenize(s))
        if len(st) < 3:
            continue
        claims += 1
        if len(st & retrieval_tokens) >= min_overlap:
            supported += 1

    if claims == 0:
        return 1.0, 0, 0

    return supported / claims, claims - supported, claims


def build_context(docs) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(
            f"""[Source {i}]
Title: {doc.metadata.get("title")}
Section: {doc.metadata.get("section_title")}
Procedure code: {doc.metadata.get("procedure_code")}
URL: {doc.metadata.get("source_url")}

{doc.page_content}
"""
        )
    return "\n\n".join(parts)


class Answerer:
    def __init__(self):
        self.model = OLLAMA_MODEL

    def answer(self, question: str, docs):
        # Conservative pre-check for out-of-scope/underspecified queries or weak evidence.
        overlap = retrieval_overlap_ratio(question, docs)
        if detect_ambiguous_or_oos_query(question) and overlap < 0.45:
            return UNCERTAINTY_RESPONSE
        if overlap < 0.20:
            return UNCERTAINTY_RESPONSE

        context = build_context(docs)

        user_prompt = f"""Question:
{question}

Context:
{context}
"""

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response["message"]["content"]

        # Post-check: if generated answer looks weakly grounded, fail safe.
        grounded_ratio, unsupported_count, claim_count = grounded_sentence_ratio(answer, docs)
        if claim_count > 0 and grounded_ratio < 0.50 and unsupported_count >= 2:
            return UNCERTAINTY_RESPONSE

        if detect_ambiguous_or_oos_query(question) and not detect_uncertainty(answer) and grounded_ratio < 0.70:
            return UNCERTAINTY_RESPONSE

        return answer