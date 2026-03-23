import json
from pathlib import Path
from typing import List

from rag.schemas import CanonicalDocument, DocumentSection, ChunkRecord


DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120


def load_canonical_docs(docs_dir: str) -> List[CanonicalDocument]:
    docs_path = Path(docs_dir)
    documents: List[CanonicalDocument] = []

    for json_path in sorted(docs_path.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sections = [
            DocumentSection(
                section_title=s["section_title"],
                section_type=s["section_type"],
                content=s["content"],
                section_order=s["section_order"],
                is_indexable=s.get("is_indexable", True),
            )
            for s in data.get("sections", [])
        ]

        doc = CanonicalDocument(
            doc_id=data["doc_id"],
            file_name=data["file_name"],
            raw_source_url=data.get("raw_source_url"),
            resolved_source_url=data.get("resolved_source_url"),
            source_domain=data.get("source_domain"),
            title=data["title"],
            institution=data.get("institution"),
            language=data.get("language"),
            document_type=data.get("document_type"),
            procedure_code=data.get("procedure_code"),
            form_ids=data.get("form_ids", []),
            fee_codes=data.get("fee_codes", []),
            raw_text=data.get("raw_text", ""),
            clean_text=data.get("clean_text", ""),
            sections=sections,
        )
        documents.append(doc)

    return documents


def build_chunk_text(doc: CanonicalDocument, section: DocumentSection, content: str) -> str:
    parts = [
        f"Title: {doc.title}",
        f"Section: {section.section_title}",
        "",
        content.strip(),
    ]
    return "\n".join(parts).strip()


def split_text_with_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # try not to cut mid-sentence/mid-word when possible
        if end < text_len:
            newline_break = text.rfind("\n", start, end)
            period_break = text.rfind(". ", start, end)
            space_break = text.rfind(" ", start, end)

            best_break = max(newline_break, period_break, space_break)
            if best_break > start + int(chunk_size * 0.6):
                end = best_break + (2 if best_break == period_break else 1)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - chunk_overlap, 0)

    return chunks


def chunk_section(doc: CanonicalDocument, section: DocumentSection,
                  chunk_size: int = DEFAULT_CHUNK_SIZE,
                  chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[ChunkRecord]:
    if not section.is_indexable:
        return []

    section_content = section.content.strip()
    if not section_content:
        return []

    raw_chunks = split_text_with_overlap(section_content, chunk_size, chunk_overlap)
    records: List[ChunkRecord] = []

    for chunk_order, raw_chunk in enumerate(raw_chunks):
        full_text = build_chunk_text(doc, section, raw_chunk)
        chunk_id = f"{doc.doc_id}__s{section.section_order}__c{chunk_order}"

        records.append(
            ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                file_name=doc.file_name,
                title=doc.title,
                source_url=doc.resolved_source_url or doc.raw_source_url,
                source_domain=doc.source_domain,
                institution=doc.institution,
                document_type=doc.document_type,
                procedure_code=doc.procedure_code,
                form_ids=doc.form_ids,
                fee_codes=doc.fee_codes,
                section_title=section.section_title,
                section_type=section.section_type,
                section_order=section.section_order,
                chunk_order=chunk_order,
                text=full_text,
                text_length=len(full_text),
            )
        )

    return records


def chunk_document(doc: CanonicalDocument,
                   chunk_size: int = DEFAULT_CHUNK_SIZE,
                   chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[ChunkRecord]:
    chunks: List[ChunkRecord] = []

    for section in doc.sections:
        chunks.extend(
            chunk_section(
                doc=doc,
                section=section,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return chunks


def chunk_documents(documents: List[CanonicalDocument],
                    chunk_size: int = DEFAULT_CHUNK_SIZE,
                    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[ChunkRecord]:
    all_chunks: List[ChunkRecord] = []

    for doc in documents:
        all_chunks.extend(
            chunk_document(
                doc=doc,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )

    return all_chunks