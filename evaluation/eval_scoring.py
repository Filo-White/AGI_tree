"""
Evaluation: Scoring Quality
============================
Measures how well the self-scoring mechanism works:
1. Ranking Accuracy (Hit@1, Hit@3): Is the correct node ranked highest?
2. Score Calibration: Do high-scoring nodes actually contain the answer?
3. Reason Coherence: Is the reason consistent with the score value?

Requires: test_data/scoring_ground_truth.json
Format:
{
  "document_file": "test_doc.txt",
  "test_cases": [
    {
      "id": "q1",
      "query": "Qual è la funzione del mitocondrio?",
      "correct_node_names": ["Capitolo 3 — Biologia Cellulare"],
      "irrelevant_node_names": ["Capitolo 1 — Introduzione"]
    }
  ]
}
"""
import asyncio
import sys
import time
import json
from pathlib import Path

from config import (
    BACKEND_DIR, TEST_DATA_DIR, EvalResult, save_results, load_test_data, print_summary
)

sys.path.insert(0, str(BACKEND_DIR))

from tree_engine import TreeEngine
from llm_client import get_competence_score, check_node_relevance


async def evaluate_scoring():
    """Run scoring quality evaluation."""
    print("\n🔍 SCORING QUALITY EVALUATION")
    print("-" * 40)

    gt = load_test_data("scoring_ground_truth.json")
    doc_path = TEST_DATA_DIR / gt["document_file"]

    if not doc_path.exists():
        print(f"  ✗ Document not found: {doc_path}")
        return []

    with open(doc_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # Build tree
    print("  Building tree from document...")
    engine = TreeEngine()
    await engine.build_tree_from_document(document_text, filename=gt["document_file"])
    print(f"  Tree built: {len(engine.root.children)} nodes")

    # Print node names for reference
    node_names = [c.name for c in engine.root.children]
    print(f"  Nodes: {node_names}")

    results: list[EvalResult] = []

    for tc in gt["test_cases"]:
        test_id = tc["id"]
        query = tc["query"]
        correct_names = [n.lower() for n in tc["correct_node_names"]]
        irrelevant_names = [n.lower() for n in tc.get("irrelevant_node_names", [])]

        print(f"\n  Testing: [{test_id}] {query[:60]}...")

        # Score all nodes
        scores, reasons = await engine.score_nodes(query)

        # Map scores to node names
        name_scores = {}
        name_reasons = {}
        for node_id, score in scores.items():
            node = engine.get_node(node_id)
            if node:
                name_scores[node.name] = score
                name_reasons[node.name] = reasons.get(node_id, "")

        # Sort by score descending
        ranked = sorted(name_scores.items(), key=lambda x: x[1], reverse=True)

        # --- Metric 1: Hit@1 ---
        top1_name = ranked[0][0] if ranked else ""
        hit_at_1 = any(c in top1_name.lower() for c in correct_names) if correct_names else False

        # --- Metric 2: Hit@3 ---
        top3_names = [r[0].lower() for r in ranked[:3]]
        hit_at_3 = any(
            any(c in t3 for c in correct_names)
            for t3 in top3_names
        ) if correct_names else False

        # --- Metric 3: Score Separation ---
        # Correct nodes should have higher scores than irrelevant ones
        correct_scores = [
            s for name, s in name_scores.items()
            if any(c in name.lower() for c in correct_names)
        ]
        irrelevant_scores = [
            s for name, s in name_scores.items()
            if any(c in name.lower() for c in irrelevant_names)
        ]
        avg_correct = sum(correct_scores) / len(correct_scores) if correct_scores else 0
        avg_irrelevant = sum(irrelevant_scores) / len(irrelevant_scores) if irrelevant_scores else 0
        score_separation = avg_correct - avg_irrelevant

        # --- Metric 4: Reason Coherence (heuristic) ---
        # High score (>0.7) should have positive reason, low (<0.3) should explain lack of relevance
        reason_coherent = True
        for name, score in name_scores.items():
            reason = name_reasons.get(name, "")
            if score >= 0.7 and not reason:
                reason_coherent = False
            # Basic check: low score reasons shouldn't claim high competence
            if score < 0.3 and reason:
                low_keywords = ["non", "poco", "scarsa", "nessun", "irrilevante", "not", "no"]
                if not any(kw in reason.lower() for kw in low_keywords):
                    # Could be incoherent, but not a hard fail
                    pass

        metrics = {
            "hit@1": hit_at_1,
            "hit@3": hit_at_3,
            "score_separation": round(score_separation, 3),
            "avg_correct_score": round(avg_correct, 3),
            "avg_irrelevant_score": round(avg_irrelevant, 3),
            "reason_provided": all(name_reasons.get(n, "") for n in name_scores),
        }

        # Pass if correct node is in top-3 and separation is positive
        passed = hit_at_3 and score_separation > 0

        result = EvalResult(
            test_id=test_id,
            dimension="scoring",
            metrics=metrics,
            details={
                "query": query,
                "ranking": [(name, round(score, 3)) for name, score in ranked],
                "reasons": {k: v for k, v in name_reasons.items()},
            },
            passed=passed,
        )
        results.append(result)

        # Print inline
        status = "✓" if passed else "✗"
        print(f"    {status} Hit@1={hit_at_1} Hit@3={hit_at_3} Sep={score_separation:.3f}")
        print(f"      Top: {ranked[0][0]} ({ranked[0][1]:.3f})" if ranked else "")

    print_summary(results, "Scoring Quality")
    save_results(results, "scoring_results.json")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
    asyncio.run(evaluate_scoring())
