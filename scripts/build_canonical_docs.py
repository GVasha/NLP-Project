import json
import sys
from pathlib import Path

# Add the parent directory to sys.path to import rag package
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.loaders.txt_loader import load_documents
from rag.processors.cleaner import clean_document_text
from rag.processors.metadata_extractor import enrich_document_metadata
from rag.processors.section_parser import split_into_sections


DATA_DIR = "Data"
OUTPUT_DIR = Path("storage/processed/docs")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} documents")

    for doc in documents:
        doc.clean_text = clean_document_text(doc.raw_text)
        doc.sections = split_into_sections(doc.clean_text)
        doc = enrich_document_metadata(doc)

        output_path = OUTPUT_DIR / f"{doc.doc_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)

        print(
            f"Saved {doc.file_name} -> {output_path.name} "
            f"| sections={len(doc.sections)} "
            f"| forms={doc.form_ids} "
            f"| procedure_code={doc.procedure_code}"
        )


if __name__ == "__main__":
    main()