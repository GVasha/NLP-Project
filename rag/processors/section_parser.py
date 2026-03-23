import re
from rag.schemas import DocumentSection


SECTION_TYPE_MAP = {
    "requirements and conditions": ("requirements", True),
    "classes and typologies": ("classification", True),
    "who are the stakeholders?": ("stakeholders", True),
    "duration of the extension": ("duration_or_validity", True),
    "how to carry out the procedure": ("procedure_steps", True),
    "documentation": ("documentation", True),

    "information": ("information", True),
    "basic data of the procedure:": ("information", True),
    "resolution of the procedure:": ("information", True),
    "bodies competent to resolve this procedure:": ("competent_body", True),
    "regulations governing administrative silence:": ("procedure_metadata", True),
    "period of validity": ("duration_or_validity", True),
    "validity period": ("duration_or_validity", True),

    "resources that can be interposed:": ("appeals", True),
    "organ that resolves the resource:": ("appeals", True),
    "regulations:": ("regulations", True),
    "code of the procedure:": ("procedure_metadata", True),
    "process": ("process", True),
    "where can i process my application?": ("where_to_apply", True),
    "models of forms": ("forms", True),
    "fee, amount and place of payment": ("fees", True),

    "all the procedures": ("portal_navigation", False),
    "all foreigner procedures and procedures": ("portal_navigation", False),
    "more information": ("related_links", False),
}


INFO_SUBHEADINGS_TO_INLINE = {
    "basic data of the procedure:",
    "resolution of the procedure:",
}


def normalize_heading(line: str) -> str:
    s = line.strip()
    s = re.sub(r"\s*\(https?://[^)]+\)\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False

    key = normalize_heading(s)

    if key in SECTION_TYPE_MAP:
        return True

    word_count = len(key.split())
    if word_count <= 6 and (s.endswith("?") or s.endswith(":")):
        return True

    return False


def classify_section(title: str) -> tuple[str, bool]:
    key = normalize_heading(title)
    return SECTION_TYPE_MAP.get(key, ("general", True))


def split_into_sections(text: str) -> list[DocumentSection]:
    lines = [line.strip() for line in text.splitlines()]

    sections = []
    current_title = "General"
    current_lines = []
    order = 0

    for line in lines:
        if not line:
            if current_lines and current_lines[-1] != "":
                current_lines.append("")
            continue

        if is_heading(line):
            new_key = normalize_heading(line)
            current_key = normalize_heading(current_title)

            # Special case:
            # If we're inside Information and hit a lightweight info subheading,
            # keep it inside the Information section instead of splitting.
            if current_key == "information" and new_key in INFO_SUBHEADINGS_TO_INLINE:
                current_lines.append(line)
                continue

            if current_lines:
                section_type, is_indexable = classify_section(current_title)
                sections.append(
                    DocumentSection(
                        section_title=current_title,
                        section_type=section_type,
                        content="\n".join(current_lines).strip(),
                        section_order=order,
                        is_indexable=is_indexable,
                    )
                )
                order += 1

            current_title = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        section_type, is_indexable = classify_section(current_title)
        sections.append(
            DocumentSection(
                section_title=current_title,
                section_type=section_type,
                content="\n".join(current_lines).strip(),
                section_order=order,
                is_indexable=is_indexable,
            )
        )

    return sections