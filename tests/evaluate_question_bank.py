import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.retrieval.smart_retriever import SmartRetriever


def normalize(text: str) -> str:
    return (text or "").strip().lower()


def parse_keywords(raw: str) -> List[str]:
    return [k.strip().lower() for k in (raw or "").split("|") if k.strip()]


def parse_expected_behavior(raw: str) -> str:
    v = (raw or "").strip().lower()
    if v in {"answer", "uncertain", "refuse"}:
        return v
    return ""


def keyword_coverage(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    hay = normalize(text)
    hits = sum(1 for kw in keywords if kw in hay)
    return hits / len(keywords)


UNCERTAINTY_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bnot (available|provided|clear|specified)\b",
        r"\bnot enough (information|context|evidence)\b",
        r"\binsufficient (information|context|evidence)\b",
        r"\bcannot (determine|confirm|answer|find)\b",
        r"\bcan\'?t (determine|confirm|answer|find)\b",
        r"\bdo not know\b",
        r"\bdon\'?t know\b",
        r"\boutside (the )?(provided|retrieved) context\b",
        r"\bno relevant (information|source)\b",
    ]
]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "this",
    "these",
    "those",
    "you",
    "your",
}


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9-]+", normalize(text))
    return [t for t in tokens if len(t) > 2 and t not in STOPWORDS]


def detect_uncertainty(answer: str) -> bool:
    if not answer:
        return False
    return any(p.search(answer) for p in UNCERTAINTY_PATTERNS)


def grounded_sentence_ratio(answer: str, retrieval_text: str, min_overlap: int = 2) -> tuple[float, int, int]:
    sentences = split_sentences(answer)
    retrieval_tokens = set(tokenize(retrieval_text))

    claim_total = 0
    supported = 0

    for s in sentences:
        # Treat explicit uncertainty/refusal statements as non-hallucinatory behavior.
        if detect_uncertainty(s):
            continue

        claim_tokens = set(tokenize(s))
        if len(claim_tokens) < 3:
            continue

        claim_total += 1
        overlap = len(claim_tokens & retrieval_tokens)
        if overlap >= min_overlap:
            supported += 1

    if claim_total == 0:
        return 1.0, 0, 0

    ratio = supported / claim_total
    return ratio, claim_total - supported, claim_total


def assess_robustness(expected_behavior: str, is_uncertain: bool, answer_cov: float, grounded_ratio: float) -> bool | None:
    if not expected_behavior:
        return None

    if expected_behavior in {"uncertain", "refuse"}:
        return is_uncertain

    # expected_behavior == "answer"
    return (not is_uncertain) and (answer_cov >= 0.35) and (grounded_ratio >= 0.50)


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
        expected_behavior = parse_expected_behavior(row.get("expected_behavior", ""))

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
        answer_cov_value = 0.0
        claim_grounded_ratio = ""
        unsupported_sentences = ""
        claim_sentences = ""
        detected_uncertainty = ""
        robustness_pass = ""
        accuracy_pass = ""
        if qa_service is not None:
            try:
                answer = qa_service.answerer.answer(question, docs)
                answer_cov_value = keyword_coverage(answer, expected_keywords)
                answer_keyword_cov = f"{answer_cov_value:.3f}"

                grounded_ratio, unsupported_count, total_claims = grounded_sentence_ratio(answer, retrieval_text)
                claim_grounded_ratio = f"{grounded_ratio:.3f}"
                unsupported_sentences = str(unsupported_count)
                claim_sentences = str(total_claims)

                uncertain = detect_uncertainty(answer)
                detected_uncertainty = str(uncertain)

                if expected_keywords:
                    accuracy_pass = str((answer_cov_value >= 0.50) and (grounded_ratio >= 0.50))
                else:
                    accuracy_pass = str(grounded_ratio >= 0.50)

                robust = assess_robustness(
                    expected_behavior=expected_behavior,
                    is_uncertain=uncertain,
                    answer_cov=answer_cov_value,
                    grounded_ratio=grounded_ratio,
                )
                robustness_pass = "" if robust is None else str(robust)
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
                "expected_behavior": expected_behavior,
                "retrieved_codes": "|".join(retrieved_codes),
                "retrieved_sections": "|".join(retrieved_sections),
                "retrieved_form_ids": "|".join(sorted(set(retrieved_form_ids))),
                "hit_procedure_code": str(hit_code),
                "hit_section_type": str(hit_section),
                "hit_form_id": str(hit_form),
                "retrieval_keyword_coverage": f"{retrieval_keyword_cov:.3f}",
                "answer_keyword_coverage": answer_keyword_cov,
                "grounded_sentence_ratio": claim_grounded_ratio,
                "unsupported_claim_sentences": unsupported_sentences,
                "claim_sentences": claim_sentences,
                "detected_uncertainty": detected_uncertainty,
                "accuracy_pass": accuracy_pass,
                "robustness_pass": robustness_pass,
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
        valid_grounded = [
            float(r["grounded_sentence_ratio"])
            for r in results
            if r["grounded_sentence_ratio"]
        ]
        valid_unsupported = [
            int(r["unsupported_claim_sentences"])
            for r in results
            if r["unsupported_claim_sentences"]
        ]
        valid_claim_counts = [
            int(r["claim_sentences"])
            for r in results
            if r["claim_sentences"]
        ]
        valid_accuracy = [
            r["accuracy_pass"] == "True"
            for r in results
            if r["accuracy_pass"]
        ]
        valid_robustness = [
            r["robustness_pass"] == "True"
            for r in results
            if r["robustness_pass"]
        ]

        if valid_answer_cov:
            avg_answer_cov = sum(valid_answer_cov) / len(valid_answer_cov)
            print(f"Avg answer keyword coverage: {avg_answer_cov:.3f}")
        else:
            print("Avg answer keyword coverage: N/A")

        if valid_grounded:
            avg_grounded = sum(valid_grounded) / len(valid_grounded)
            print(f"Avg grounded sentence ratio: {avg_grounded:.3f}")
        else:
            print("Avg grounded sentence ratio: N/A")

        if valid_claim_counts:
            total_unsupported = sum(valid_unsupported)
            total_claims = sum(valid_claim_counts)
            hallucination_proxy = total_unsupported / total_claims if total_claims else 0.0
            print(f"Unsupported claim sentences: {total_unsupported}/{total_claims} ({100*hallucination_proxy:.1f}%)")
        else:
            print("Unsupported claim sentences: N/A")

        if valid_accuracy:
            accuracy_pass_rate = sum(valid_accuracy) / len(valid_accuracy)
            print(f"Accuracy pass rate: {sum(valid_accuracy)}/{len(valid_accuracy)} ({100*accuracy_pass_rate:.1f}%)")
        else:
            print("Accuracy pass rate: N/A")

        if valid_robustness:
            robust_pass_rate = sum(valid_robustness) / len(valid_robustness)
            print(f"Robustness pass rate: {sum(valid_robustness)}/{len(valid_robustness)} ({100*robust_pass_rate:.1f}%)")
        else:
            print("Robustness pass rate: N/A")

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
