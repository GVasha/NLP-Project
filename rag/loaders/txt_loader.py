from pathlib import Path
from typing import List, Optional
from rag.schemas import CanonicalDocument


def extract_source_url(text: str) -> Optional[str]:
    lines = text.splitlines()
    if not lines:
        return None

    first = lines[0].strip()
    if first.lower().startswith("source:"):
        return first.split(":", 1)[1].strip()
    return None


def strip_source_line(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].strip().lower().startswith("source:"):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def infer_initial_title(body_text: str, fallback: str) -> str:
    generic = {
        "foreigner procedures",
        "procedures and procedures",
        "information",
        "process",
        "documentation",
        "more information",
    }

    for line in body_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower() in generic:
            continue
        if len(line) <= 140:
            return line

    return fallback


def load_txt_file(file_path: Path) -> CanonicalDocument:
    raw = file_path.read_text(encoding="utf-8")
    raw_source_url = extract_source_url(raw)
    body = strip_source_line(raw)
    title = infer_initial_title(body, file_path.stem)

    return CanonicalDocument(
        doc_id=file_path.stem,
        file_name=file_path.name,
        raw_source_url=raw_source_url,
        resolved_source_url=raw_source_url,
        source_domain=None,
        title=title,
        institution=None,
        language=None,
        document_type=None,
        procedure_code=None,
        raw_text=body,
    )


def load_documents(data_dir: str) -> List[CanonicalDocument]:
    path = Path(data_dir)
    docs = []

    for file_path in sorted(path.glob("*.txt")):
        docs.append(load_txt_file(file_path))

    return docs