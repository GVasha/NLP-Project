from ollama import chat

from rag.config import OLLAMA_MODEL


SYSTEM_PROMPT = """You answer immigration and administrative procedure questions using only the provided context.

Rules:
- Use only the retrieved context for factual claims.
- Do not invent requirements, deadlines, offices, forms, steps, or legal advice.
- If the retrieved context does not clearly answer the question, say so plainly.
- Be concise, practical, and neutral.
- Prefer a clean structure:
  1. Direct answer
  2. Key details or limitations if needed
- Do not ask follow-up questions.
- Do not add a separate Sources section inside the answer.
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

Write the answer for the user.

Formatting instructions:
- Start with the answer immediately.
- If the context supports the answer, explain it in 1-2 short paragraphs.
- If the context is incomplete or ambiguous, explicitly say that the retrieved sources do not fully answer the question.
- Only mention details that are directly supported by the context.
"""

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        return response["message"]["content"].strip()