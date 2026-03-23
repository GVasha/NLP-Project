import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from rag.indexing.embedder import get_embeddings


def load_chunks_jsonl(chunks_path: str) -> List[dict]:
    records = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def chunk_record_to_langchain_document(record: dict) -> Document:
    metadata = {
        "chunk_id": record["chunk_id"],
        "doc_id": record["doc_id"],
        "file_name": record["file_name"],
        "title": record["title"],
        "source_url": record.get("source_url"),
        "source_domain": record.get("source_domain"),
        "institution": record.get("institution"),
        "document_type": record.get("document_type"),
        "procedure_code": record.get("procedure_code"),
        "form_ids": record.get("form_ids", []),
        "fee_codes": record.get("fee_codes", []),
        "section_title": record["section_title"],
        "section_type": record["section_type"],
        "section_order": record["section_order"],
        "chunk_order": record["chunk_order"],
        "text_length": record["text_length"],
    }

    return Document(
        page_content=record["text"],
        metadata=metadata,
    )


def build_faiss_index_from_chunks(chunks_path: str, model_name: str):
    chunk_records = load_chunks_jsonl(chunks_path)
    documents = [chunk_record_to_langchain_document(r) for r in chunk_records]

    embeddings = get_embeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore, len(documents)


def save_faiss_index(vectorstore: FAISS, output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(output_path))


def load_faiss_index(index_dir: str, model_name: str) -> FAISS:
    embeddings = get_embeddings(model_name=model_name)
    return FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )