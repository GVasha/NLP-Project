import re


NOISE_EXACT = {
    "See all",
    "Open a new window.",
    "External link.",
}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_html_artifacts(text: str) -> str:
    text = text.replace("<strong", "")
    text = text.replace("</strong>", "")
    return text


def remove_light_noise(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        s = line.strip()

        if s in NOISE_EXACT:
            continue

        if s.lower() in {
            "opens new window",
            "opens in a new window",
            "external link",
        }:
            continue

        cleaned_lines.append(s)

    return "\n".join(cleaned_lines).strip()


def clean_document_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = fix_html_artifacts(text)
    text = remove_light_noise(text)
    text = normalize_whitespace(text)
    return text