from rag.generation.answerer import Answerer
from rag.retrieval.smart_retriever import SmartRetriever


class QAService:
    def __init__(self):
        self.retriever = SmartRetriever()
        self.answerer = Answerer()

    def ask(self, question: str) -> dict:
        docs = self.retriever.search(question)
        answer = self.answerer.answer(question, docs)

        sources = [
            {
                "title": d.metadata.get("title"),
                "section": d.metadata.get("section_title"),
                "procedure_code": d.metadata.get("procedure_code"),
                "source_url": d.metadata.get("source_url"),
            }
            for d in docs
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }