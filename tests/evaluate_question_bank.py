import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieval.smart_retriever import SmartRetriever


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def parse_keywords(raw: str) -> List[str]:
    return [k.strip().lower() for k in (raw or "").split("|") if k.strip()]


def keyword_coverage(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    hay = normalize(text)
    hits = sum(1 for kw in keywords if kw in hay)
    return hits / len(keywords)


def run_eval(question_bank_path: Path, output_path: Path, top_k: int, with_answers: bool) -> None:
    retriever = SmartRetriever()

    qa_service = None
    if with_answers:
        from rag.service.qa_service import QAService

        qa_service = QAService()

    rows: List[Dict[str, str]] = []
    with question_bank_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []

    for row in rows:
        qid = row["id"]
        question = row["question"]
        expected_keywords = parse_keywords(row.get("expected_keywords", ""))
        expected_code = row.get("expected_procedure_code", "").strip()
        expected_section = row.get("expected_section_type", "").strip()
        expected_form_id = row.get("expected_form_id", "").strip().upper()

        docs = retriever.search(question, k_final=top_k)

        retrieved_codes = [str(d.metadata.get("procedure_code", "")) for d in docs]
        retrieved_sections = [str(d.metadata.get("section_type", "")) for d in docs]
        retrieved_form_ids = []
        for d in docs:
            form_ids = d.metadata.get("form_ids", []) or []
            retrieved_form_ids.extend([str(fid).upper() for fid in form_ids])

        retrieval_text = "\n".join(d.page_content for d in docs)

        hit_code = expected_code in retrieved_codes if expected_code else True
        hit_section = expected_section in retrieved_sections if expected_section else True
        hit_form = (expected_form_id in retrieved_form_ids) if expected_form_id else True
        retrieval_keyword_cov = keyword_coverage(retrieval_text, expected_keywords)

        answer = ""
        answer_keyword_cov = ""
        if qa_service is not None:
            try:
                answer = qa_service.answerer.answer(question, docs)
                answer_keyword_cov = f"{keyword_coverage(answer, expected_keywords):.3f}"
            except Exception as ex:
                answer = f"ERROR: {type(ex).__name__}: {ex}"
                answer_keyword_cov = ""

        results.append(
            {
                "id": qid,
                "question": question,
                "expected_procedure_code": expected_code,
                "expected_section_type": expected_section,
                "expected_form_id": expected_form_id,
                "retrieved_codes": "|".join(retrieved_codes),
                "retrieved_sections": "|".join(retrieved_sections),
                "retrieved_form_ids": "|".join(sorted(set(retrieved_form_ids))),
                "hit_procedure_code": str(hit_code),
                "hit_section_type": str(hit_section),
                "hit_form_id": str(hit_form),
                "retrieval_keyword_coverage": f"{retrieval_keyword_cov:.3f}",
                "answer_keyword_coverage": answer_keyword_cov,
                "answer_preview": (answer[:220] + "...") if len(answer) > 220 else answer,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(results[0].keys()) if results else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    code_hits = sum(r["hit_procedure_code"] == "True" for r in results)
    section_hits = sum(r["hit_section_type"] == "True" for r in results)
    form_hits = sum(r["hit_form_id"] == "True" for r in results)
    retrieval_cov_avg = sum(float(r["retrieval_keyword_coverage"]) for r in results) / total if total else 0.0

    print(f"Question bank: {question_bank_path}")
    print(f"Rows: {total}")
    print(f"Top-k evaluated: {top_k}")
    print(f"Procedure hit@{top_k}: {code_hits}/{total} ({(100*code_hits/total):.1f}%)")
    print(f"Section hit@{top_k}: {section_hits}/{total} ({(100*section_hits/total):.1f}%)")
    print(f"Form hit@{top_k}: {form_hits}/{total} ({(100*form_hits/total):.1f}%)")
    print(f"Avg retrieval keyword coverage: {retrieval_cov_avg:.3f}")

    if with_answers:
        valid_answer_cov = [
            float(r["answer_keyword_coverage"])
            for r in results
            if r["answer_keyword_coverage"]
        ]
        if valid_answer_cov:
            avg_answer_cov = sum(valid_answer_cov) / len(valid_answer_cov)
            print(f"Avg answer keyword coverage: {avg_answer_cov:.3f}")
        else:
            print("Avg answer keyword coverage: N/A")

    print(f"Saved detailed results: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval/answers with a question bank CSV.")
    parser.add_argument(
        "--question-bank",
        default="tests/questionBank.csv",
        help="Path to the question bank CSV",
    )
    parser.add_argument(
        "--output",
        default="tests/eval_results_v1.csv",
        help="Path to output CSV with detailed results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of final retrieved chunks to evaluate",
    )
    parser.add_argument(
        "--with-answers",
        action="store_true",
        help="Also generate answers and compute answer keyword coverage",
    )

    args = parser.parse_args()
    run_eval(
        question_bank_path=Path(args.question_bank),
        output_path=Path(args.output),
        top_k=args.top_k,
        with_answers=args.with_answers,
    )


if __name__ == "__main__":
    main()
