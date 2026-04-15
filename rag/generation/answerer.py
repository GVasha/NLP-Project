import json
import urllib.error
import urllib.request

from rag.config import OLLAMA_MODEL


SYSTEM_PROMPT = """You answer immigration/admin procedure questions using only the provided context.
Rules:
- Use only the retrieved context.
- If the answer is not clearly in the context, say that the information is not available in the retrieved sources.
- Be concise and practical.
- End with a short Sources section listing title + section + procedure code.
"""


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

    def answer(self, question: str, docs):
        context = build_context(docs)

        user_prompt = f"""Question:
{question}

Context:
{context}
"""

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
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
        return parsed["message"]["content"]