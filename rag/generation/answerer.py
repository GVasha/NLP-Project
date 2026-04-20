import json
import urllib.error
import urllib.request
import re

from rag.config import OLLAMA_MODEL
from rag.retrieval.smart_retriever import detect_intents, extract_form_id, extract_procedure_code


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

DOMAIN_HINTS = {
    "nie", "ex", "form", "forms", "fee", "fees", "790", "012", "procedure",
    "residence", "card", "authorization", "student", "foreigner", "office", "police",
    "apply", "application", "documents", "passport", "appeal", "regulation",
}


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9-]+", normalize(text))
    return [t for t in tokens if len(t) > 2 and t not in QUERY_STOPWORDS]


GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy|greetings|what'?s\s+up|sup)\b",
    re.IGNORECASE,
)

GREETING_RESPONSE = (
    "Hello! I'm here to help with immigration and administrative procedure questions. "
    "Feel free to ask about visas, residence cards, NIE, required documents, fees, or any other procedure."
)


def is_greeting(question: str) -> bool:
    return bool(GREETING_PATTERNS.match(question.strip())) and len(question.strip().split()) <= 6


def detect_ambiguous_or_oos_query(question: str) -> bool:
    q = normalize(question)
    if any(x in q for x in [
        "weather", "rent", "exchange rate", "whatsapp", "book my appointment",
        "guarantee", "guaranteed", "internal confidence", "faster city", "private email",
        "submit my application for me", "court strategy", "same-day premium",
    ]):
        return True

    q_tokens = set(tokenize(q))
    has_domain_hint = any(t in DOMAIN_HINTS for t in q_tokens)

    # Very short generic questions are often under-specified and should be handled conservatively.
    if len(q_tokens) <= 3 and not has_domain_hint and any(x in q for x in ["documents", "apply", "where", "what do i need"]):
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


def split_candidate_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    out = []
    for s in chunks:
        s = s.strip(" -\t")
        if len(s) < 25:
            continue
        if len(s) > 320:
            continue
        out.append(s)
    return out


def intent_keywords(intents: set[str]) -> set[str]:
    k = set()
    if "forms" in intents:
        k |= {"ex", "form", "forms", "model"}
    if "fees" in intents:
        k |= {"fee", "tax", "790", "012", "payment"}
    if "where_to_apply" in intents:
        k |= {"office", "police", "station", "apply", "submit"}
    if "documentation" in intents:
        k |= {"document", "documents", "passport", "proof", "copy", "original"}
    if "duration_or_validity" in intents:
        k |= {"validity", "duration", "days", "months", "renewable", "period"}
    if "procedure_metadata" in intents:
        k |= {"procedure", "code"}
    if "appeals" in intents:
        k |= {"appeal", "resource"}
    if "competent_body" in intents:
        k |= {"competent", "body", "resolves", "authority"}
    if "regulations" in intents:
        k |= {"regulation", "law", "normative"}
    if "process" in intents or "procedure_steps" in intents:
        k |= {"process", "steps", "carry", "procedure"}
    return k


def select_support_sentences(question: str, docs, intents: set[str], limit: int = 4) -> list[str]:
    q_tokens = set(tokenize(question))
    i_tokens = intent_keywords(intents)

    candidates: list[tuple[int, str]] = []
    seen = set()
    for d in docs:
        for s in split_candidate_sentences(d.page_content):
            key = normalize(s)
            if key in seen:
                continue
            seen.add(key)
            st = set(tokenize(s))
            if not st:
                continue
            overlap_q = len(st & q_tokens)
            overlap_i = len(st & i_tokens)
            score = overlap_q * 3 + overlap_i * 2
            if score <= 0:
                continue
            candidates.append((score, s))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in candidates[:limit]]


def build_sources_block(docs) -> str:
    lines = []
    for d in docs:
        lines.append(
            f"- {d.metadata.get('title')} | {d.metadata.get('section_title')} | {d.metadata.get('procedure_code')}"
        )
    return "\n".join(lines)


def build_extractive_answer(question: str, docs) -> str:
    intents = detect_intents(question)
    explicit_proc = extract_procedure_code(question)
    explicit_form = extract_form_id(question)

    lines = []

    if explicit_proc:
        hits = [d for d in docs if str(d.metadata.get("procedure_code") or "") == explicit_proc]
        if not hits:
            return UNCERTAINTY_RESPONSE

    if explicit_form:
        has_form = any(
            explicit_form in [str(x).upper() for x in (d.metadata.get("form_ids") or [])]
            or explicit_form in (d.page_content or "").upper()
            for d in docs
        )
        if not has_form:
            return UNCERTAINTY_RESPONSE

    # For form/code questions, provide high-precision direct answers from metadata first.
    if "procedure_metadata" in intents:
        code = next((str(d.metadata.get("procedure_code") or "") for d in docs if d.metadata.get("procedure_code")), "")
        if code:
            lines.append(f"Procedure code: {code}.")

    if "forms" in intents:
        form_candidates = []
        for d in docs:
            for fid in d.metadata.get("form_ids") or []:
                fid_u = str(fid).upper()
                if fid_u not in form_candidates:
                    form_candidates.append(fid_u)
        if form_candidates:
            lines.append("Relevant form(s): " + ", ".join(form_candidates[:3]) + ".")

    support = select_support_sentences(question, docs, intents, limit=4)
    lines.extend(support)

    if not lines:
        return UNCERTAINTY_RESPONSE

    body = "\n".join(f"- {line}" for line in lines[:5])
    return f"Based on the retrieved sources:\n{body}\n\nSources:\n{build_sources_block(docs)}"


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
        self.ollama_url = "http://127.0.0.1:11434/api/chat"

    def answer(self, question: str, docs, history: list[dict] | None = None):
        has_history = bool(history)

        if is_greeting(question) and not has_history:
            return GREETING_RESPONSE

        # Only block queries that are explicitly out-of-scope (weather, etc.).
        # Do not block on vocabulary overlap — natural language questions ("how do I get
        # my visa?") won't share tokens with technical docs, but are still valid queries.
        if detect_ambiguous_or_oos_query(question) and not has_history:
            return UNCERTAINTY_RESPONSE

        context = build_context(docs)

        user_prompt = f"""Question:
{question}

Context:
{context}
"""

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in (history or []):
            role = turn.get("role", "user")
            text = str(turn.get("text", "")).strip()
            if role in {"user", "assistant"} and text:
                messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        request = urllib.request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8")
            except Exception:  # noqa: BLE001
                details = str(exc)
            raise RuntimeError(f"Ollama request failed: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Could not reach Ollama at http://127.0.0.1:11434. "
                "Start Ollama and verify the service is running."
            ) from exc

        parsed = json.loads(raw)
        answer = parsed["message"]["content"]

        return answer
