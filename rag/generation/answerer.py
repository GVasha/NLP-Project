from ollama import chat

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

    def answer(self, question: str, docs):
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

        return response["message"]["content"]