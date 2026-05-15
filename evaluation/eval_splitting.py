"""
Evaluation: Document Splitting Quality
========================================
Measures how well the system splits a document into nodes:
1. Classification Accuracy: Is the document type correctly identified?
2. Node Count Accuracy: |expected - actual| / expected
3. Name Coverage: % of expected sections found (fuzzy match)
4. Text Coverage: % of total document text covered by nodes
5. No Empty Nodes: All nodes should have non-trivial content

Requires: test_data/splitting_ground_truth.json
Format:
{
  "test_cases": [
    {
      "id": "doc1",
      "document_file": "test_book.txt",
      "expected_doc_type": "book",
      "expected_sections": [
        "Introduzione",
        "Capitolo 1 — Fondamenti",
        "Capitolo 2 — Applicazioni",
        "Conclusioni"
      ],
      "min_nodes": 3,
      "max_nodes": 8
    }
  ]
}
"""
import asyncio
import sys
from pathlib import Path
from difflib import SequenceMatcher

from config import (
    BACKEND_DIR, TEST_DATA_DIR, EvalResult, save_results, load_test_data, print_summary
)

sys.path.insert(0, str(BACKEND_DIR))

from tree_engine import TreeEngine
from llm_client import classify_document, detect_top_level_nodes


def fuzzy_match(name: str, candidates: list[str], threshold: float = 0.55) -> str | None:
    """Find best fuzzy match for a name in a list of candidates."""
    best_score = 0
    best_match = None
    name_lower = name.lower().strip()
    for candidate in candidates:
        cand_lower = candidate.lower().strip()
        # Direct substring check
        if name_lower in cand_lower or cand_lower in name_lower:
            return candidate
        # SequenceMatcher ratio
        ratio = SequenceMatcher(None, name_lower, cand_lower).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = candidate
    return best_match if best_score >= threshold else None


async def evaluate_splitting():
    """Run document splitting quality evaluation."""
    print("\n📐 DOCUMENT SPLITTING QUALITY EVALUATION")
    print("-" * 40)

    gt = load_test_data("splitting_ground_truth.json")
    results: list[EvalResult] = []

    for tc in gt["test_cases"]:
        test_id = tc["id"]
        doc_path = TEST_DATA_DIR / tc["document_file"]

        if not doc_path.exists():
            print(f"  ✗ [{test_id}] Document not found: {doc_path}")
            results.append(EvalResult(
                test_id=test_id, dimension="splitting",
                metrics={"error": "file_not_found"}, passed=False
            ))
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            document_text = f.read()

        total_chars = len(document_text)
        print(f"\n  Testing: [{test_id}] {tc['document_file']} ({total_chars} chars)")

        # --- Step 1: Classification ---
        classification = await classify_document(document_text)
        detected_type = classification.get("doc_type", "unknown")
        expected_type = tc["expected_doc_type"]
        type_correct = detected_type == expected_type

        print(f"    Classification: {detected_type} (expected: {expected_type}) {'✓' if type_correct else '✗'}")

        # --- Step 2: Node Detection ---
        engine = TreeEngine()
        await engine.build_tree_from_document(document_text, filename=tc["document_file"])

        actual_nodes = engine.root.children
        actual_names = [n.name for n in actual_nodes]
        expected_sections = tc["expected_sections"]
        min_nodes = tc.get("min_nodes", 2)
        max_nodes = tc.get("max_nodes", 20)

        print(f"    Nodes detected: {len(actual_names)} (expected range: {min_nodes}-{max_nodes})")
        print(f"    Names: {actual_names}")

        # --- Metric 1: Node Count in Range ---
        count_in_range = min_nodes <= len(actual_names) <= max_nodes

        # --- Metric 2: Node Count Error ---
        expected_count = len(expected_sections)
        count_error = abs(len(actual_names) - expected_count) / max(expected_count, 1)

        # --- Metric 3: Name Coverage (expected sections found) ---
        matched_expected = 0
        match_details = {}
        for expected_name in expected_sections:
            match = fuzzy_match(expected_name, actual_names)
            if match:
                matched_expected += 1
                match_details[expected_name] = match
            else:
                match_details[expected_name] = None

        name_coverage = matched_expected / len(expected_sections) if expected_sections else 0

        # --- Metric 4: Text Coverage ---
        covered_chars = 0
        for nd_data in engine._nodes_data:
            covered_chars += len(nd_data.get("text", ""))
        text_coverage = min(covered_chars / total_chars, 1.0) if total_chars > 0 else 0

        # --- Metric 5: No Empty Nodes ---
        empty_nodes = [n.name for n in actual_nodes if len(n.context.strip()) < 50]
        no_empty = len(empty_nodes) == 0

        metrics = {
            "classification_correct": type_correct,
            "detected_type": detected_type,
            "node_count": len(actual_names),
            "count_in_range": count_in_range,
            "count_error": round(count_error, 3),
            "name_coverage": round(name_coverage, 3),
            "text_coverage": round(text_coverage, 3),
            "no_empty_nodes": no_empty,
            "empty_nodes": empty_nodes,
        }

        # Pass criteria: type correct, count in range, >60% name coverage, >80% text coverage
        passed = (
            type_correct
            and count_in_range
            and name_coverage >= 0.6
            and text_coverage >= 0.8
        )

        result = EvalResult(
            test_id=test_id,
            dimension="splitting",
            metrics=metrics,
            details={
                "actual_names": actual_names,
                "expected_sections": expected_sections,
                "match_details": match_details,
                "classification": classification,
            },
            passed=passed,
        )
        results.append(result)

        status = "✓" if passed else "✗"
        print(f"    {status} Type={type_correct} Count={count_in_range} "
              f"NameCov={name_coverage:.0%} TextCov={text_coverage:.0%}")

    print_summary(results, "Document Splitting Quality")
    save_results(results, "splitting_results.json")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
    asyncio.run(evaluate_splitting())
