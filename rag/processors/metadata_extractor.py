import re
from urllib.parse import urlparse

from rag.schemas import CanonicalDocument


FORM_PATTERN = re.compile(r"\bEX\s?(\d{2})\b", re.IGNORECASE)
PROCEDURE_CODE_PATTERN = re.compile(r"\b\d{6}\b")
FEE_CODE_PATTERN = re.compile(r"\b790\b.*?\b012\b|\b012\b.*?\b790\b", re.IGNORECASE)


def infer_source_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return None


def infer_institution(domain: str | None) -> str | None:
    if not domain:
        return None
    if "policia.gob.es" in domain or "policia.es" in domain:
        return "Policía Nacional"
    if "boe.es" in domain:
        return "BOE"
    if "inclusion.gob.es" in domain:
        return "Ministerio de Inclusión"
    if "exteriores.gob.es" in domain:
        return "Ministerio de Asuntos Exteriores"
    return domain


def infer_document_type(text: str) -> str | None:
    lowered = text.lower()
    if "code of the procedure" in lowered or "how to carry out the procedure" in lowered:
        return "procedure_page"
    if "foreign identity number" in lowered or "número de identidad de extranjero" in lowered:
        return "guidance_page"
    return None


def get_core_metadata_text(doc: CanonicalDocument) -> str:
    allowed_types = {
        "general",
        "requirements",
        "classification",
        "stakeholders",
        "duration_or_validity",
        "procedure_steps",
        "documentation",
        "information",
        "procedure_metadata",
        "competent_body",
        "appeals",
        "regulations",
        "process",
        "where_to_apply",
        "forms",
        "fees",
    }

    parts = []
    for section in doc.sections:
        if section.is_indexable and section.section_type in allowed_types:
            parts.append(section.content)

    return "\n\n".join(parts)


def extract_form_ids_from_sections(doc: CanonicalDocument) -> list[str]:
    priority_types = {"forms", "documentation", "general", "procedure_steps"}
    found = []

    for section in doc.sections:
        if not section.is_indexable:
            continue
        if section.section_type not in priority_types:
            continue

        matches = FORM_PATTERN.findall(section.content)
        found.extend([f"EX{num}" for num in matches])

    return sorted(set(found))


def extract_procedure_code(text: str) -> str | None:
    # prefer 6-digit procedure codes
    matches = PROCEDURE_CODE_PATTERN.findall(text)
    for match in matches:
        if len(match) == 6:
            return match
    return None


def extract_fee_codes(text: str) -> list[str]:
    codes = []
    if FEE_CODE_PATTERN.search(text):
        codes.append("790-012")
    return codes


def enrich_document_metadata(doc: CanonicalDocument) -> CanonicalDocument:
    doc.source_domain = infer_source_domain(doc.raw_source_url)
    doc.institution = infer_institution(doc.source_domain)
    doc.document_type = infer_document_type(doc.clean_text)

    core_text = get_core_metadata_text(doc)

    doc.procedure_code = extract_procedure_code(core_text)
    doc.form_ids = extract_form_ids_from_sections(doc)
    doc.fee_codes = extract_fee_codes(core_text)

    return doc