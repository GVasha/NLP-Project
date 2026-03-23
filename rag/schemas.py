from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class DocumentSection:
    section_title: str
    section_type: str
    content: str
    section_order: int
    is_indexable: bool = True


@dataclass
class CanonicalDocument:
    doc_id: str
    file_name: str
    raw_source_url: Optional[str]
    resolved_source_url: Optional[str]
    source_domain: Optional[str]
    title: str
    institution: Optional[str]
    language: Optional[str]
    document_type: Optional[str]
    procedure_code: Optional[str]
    form_ids: List[str] = field(default_factory=list)
    fee_codes: List[str] = field(default_factory=list)
    raw_text: str = ""
    clean_text: str = ""
    sections: List[DocumentSection] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)