from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class DocumentSection:
    section_title: str
    content: str
    section_order: int


@dataclass
class CanonicalDocument:
    doc_id: str
    file_name: str
    source_url: Optional[str]
    title: str
    raw_text: str
    clean_text: str = ""
    institution: Optional[str] = None
    language: Optional[str] = None
    document_type: Optional[str] = None
    sections: List[DocumentSection] = field(default_factory=list)


@dataclass
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: Dict