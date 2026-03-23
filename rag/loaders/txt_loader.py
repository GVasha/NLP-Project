from pathlib import Path
from typing import List, Optional
import re

from rag.schemas import CanonicalDocument


def extract_source_url(text: str) -> Optional[str]:
    lines = text.splitlines()
    if not lines:
        return None

    first_line = lines[0].strip()
    if first_line.lower().startswith("source:"):
        return first_line.replace("Source:", "", 1).strip()
    return None


def remove_source_line(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("source:"):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def infer_title(text: str, fallback: str) -> str:
    """
    Heuristic:
    - skip empty lines
    - skip generic headings if possible
    - choose first meaningful short line
    """
    generic_titles = {
        "foreigner procedures",
        "procedures and procedures",
        "information",
        "process",
        "documentation",
        "more information",
    }

    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue

        lowered = clean.lower()
        if lowered in generic_titles:
            continue

        if len(clean) <= 120:
            return clean

    return fallback


def infer_doc_id(file_path: Path) -> str:
    return file_path.stem


def load_txt_file(file_path: Path) -> CanonicalDocument:
    raw_text = file_path.read_text(encoding="utf-8")
    source_url = extract_source_url(raw_text)
    body_text = remove_source_line(raw_text)
    title = infer_title(body_text, fallback=file_path.stem)

    return CanonicalDocument(
        doc_id=infer_doc_id(file_path),
        file_name=file_path.name,
        source_url=source_url,
        title=title,
        raw_text=body_text,
    )


def load_documents(data_dir: str) -> List[CanonicalDocument]:
    data_path = Path(data_dir)
    documents = []

    for file_path in sorted(data_path.glob("*.txt")):
        documents.append(load_txt_file(file_path))

    return documents