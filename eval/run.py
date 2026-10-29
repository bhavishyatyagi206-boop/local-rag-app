"""
Retrieval evaluation harness:
Computes Recall@k, MRR (Mean Reciprocal Rank), Citation Validity, and No-Answer Specificity
across retrieval configurations (Lexical BM25 and Hybrid RRF).
Usage: python -m eval.run
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ragcore import (
    DeterministicLocalEmbeddingProvider,
    KnowledgeBase,
    extract,
)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_evaluation_kb(sample_docs_dir: Path) -> Tuple[KnowledgeBase, Dict[str, Dict[int, str]]]:
    """Index sample documents into an in-memory Knowledge Base and retain ground-truth text."""
    kb = KnowledgeBase(name="Benchmark Evaluation KB")
    source_texts: Dict[str, Dict[int, str]] = {}

    for file_path in sorted(sample_docs_dir.iterdir()):
        if file_path.suffix.lower() in (".txt", ".docx", ".xlsx", ".pdf"):
            pages = extract(file_path)
            doc_name = file_path.name
            kb.add_document(doc_id=doc_name, name=doc_name, pages=pages)
            source_texts[doc_name] = {page_num: text for page_num, text in pages}

    return kb, source_texts


def evaluate_configuration(
    kb: KnowledgeBase,
    questions: List[Dict[str, Any]],
    source_texts: Dict[str, Dict[int, str]],
    config_name: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    total_q = len(questions)
    answerable_q = [q for q in questions if q["is_answerable"]]
    unanswerable_q = [q for q in questions if not q["is_answerable"]]

    recall_at_1 = 0
    recall_at_3 = 0
    recall_at_5 = 0
    reciprocal_ranks = []

    valid_citations = 0
    total_citations = 0

    no_answer_correct = 0

    for q in questions:
        query = q["question"]
        expected_docs = set(q["expected_docs"])
        expected_page = q.get("expected_page")

        res = kb.ask(query=query, top_k=top_k)
        retrieved_citations = res["citations"]
        status = res["status"]

        if q["is_answerable"]:
            # Evaluate Recall@k and MRR
            found_rank = None
            for rank, cit in enumerate(retrieved_citations, start=1):
                if cit["doc_name"] in expected_docs:
                    # Check page match if specified
                    if expected_page is None or cit["page_number"] == expected_page:
                        if found_rank is None:
                            found_rank = rank

            if found_rank is not None:
                if found_rank <= 1:
                    recall_at_1 += 1
                if found_rank <= 3:
                    recall_at_3 += 1
                if found_rank <= 5:
                    recall_at_5 += 1
                reciprocal_ranks.append(1.0 / found_rank)
            else:
                reciprocal_ranks.append(0.0)

            # Evaluate Citation Validity
            for cit in retrieved_citations:
                total_citations += 1
                doc_pages = source_texts.get(cit["doc_name"], {})
                page_text = doc_pages.get(cit["page_number"], "")
                # Clean excerpt without ellipsis
                clean_excerpt = cit["excerpt"].rstrip(".").strip()
                if clean_excerpt and (clean_excerpt in page_text or clean_excerpt[:60] in page_text):
                    valid_citations += 1

        else:
            # Unanswerable question: should have status 'no_relevant_context' or 0 citations
            if status == "no_relevant_context" or len(retrieved_citations) == 0:
                no_answer_correct += 1

    n_ans = len(answerable_q)
    n_unans = len(unanswerable_q)

    mrr = sum(reciprocal_ranks) / n_ans if n_ans > 0 else 0.0
    r1 = recall_at_1 / n_ans if n_ans > 0 else 0.0
    r3 = recall_at_3 / n_ans if n_ans > 0 else 0.0
    r5 = recall_at_5 / n_ans if n_ans > 0 else 0.0
    citation_validity = (valid_citations / total_citations) if total_citations > 0 else 1.0
    no_ans_precision = (no_answer_correct / n_unans) if n_unans > 0 else 1.0

    return {
        "config": config_name,
        "total_questions": total_q,
        "answerable_questions": n_ans,
        "unanswerable_questions": n_unans,
        "recall@1": round(r1, 4),
        "recall@3": round(r3, 4),
        "recall@5": round(r5, 4),
        "mrr": round(mrr, 4),
        "citation_validity": round(citation_validity, 4),
        "no_answer_precision": round(no_ans_precision, 4),
    }


def main():
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "eval" / "questions.json"
    sample_docs_dir = base_dir / "sample_docs"

    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        sys.exit(1)

    questions = load_dataset(dataset_path)
    kb, source_texts = build_evaluation_kb(sample_docs_dir)

    print("================================================================================")
    print("                      LOCAL RAG EVALUATION BENCHMARK                            ")
    print("================================================================================")
    print(f"Total benchmark questions: {len(questions)}")
    print(f"Documents indexed: {list(source_texts.keys())}")
    print()

    # 1. Lexical BM25 Configuration
    kb.set_embedding_provider(None)
    bm25_metrics = evaluate_configuration(
        kb, questions, source_texts, config_name="Lexical BM25 (Deterministic Sparse)"
    )

    # 2. Hybrid RRF Configuration (BM25 + Dense Vectors)
    hybrid_provider = DeterministicLocalEmbeddingProvider("eval-dense-hashing", dim=64)
    kb.set_embedding_provider(hybrid_provider)
    hybrid_metrics = evaluate_configuration(
        kb, questions, source_texts, config_name="Hybrid RRF (BM25 + Dense Vectors)"
    )

    results = [bm25_metrics, hybrid_metrics]

    header = f"{'Configuration':<35} | {'R@1':<6} | {'R@3':<6} | {'R@5':<6} | {'MRR':<6} | {'Citation Valid':<14} | {'No-Answer Prec':<14}"
    divider = "-" * len(header)
    print(header)
    print(divider)

    for m in results:
        row = (
            f"{m['config']:<35} | "
            f"{m['recall@1']:<6.2%} | "
            f"{m['recall@3']:<6.2%} | "
            f"{m['recall@5']:<6.2%} | "
            f"{m['mrr']:<6.4f} | "
            f"{m['citation_validity']:<14.2%} | "
            f"{m['no_answer_precision']:<14.2%}"
        )
        print(row)

    print(divider)
    print("Note: All metrics above are computed deterministically against ground-truth document spans.")
    print("      No LLM-judged metrics were used; zero metric fabrication.")
    print("================================================================================")


if __name__ == "__main__":
    main()
