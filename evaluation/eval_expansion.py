"""
Evaluation: Expansion Effectiveness
=====================================
Measures how well the auto-expansion mechanism works:
1. Auto-expand Precision: Does it NOT expand when the query is out-of-scope?
2. Auto-expand Recall: Does it expand when the answer requires sub-sections?
3. Relevance Check Accuracy: Does check_node_relevance return correct results?
4. Expansion Quality: Do the generated leaves make sense?

Requires: test_data/expansion_ground_truth.json
Format:
{
  "document_file": "test_doc.txt",
  "test_cases": [
    {
      "id": "exp1",
      "query": "Qual è il processo di fotosintesi?",
      "should_expand": true,
      "expected_expand_node": "Capitolo 3 — Biologia",
      "reason": "Answer is in sub-section of this chapter"
    },
    {
      "id": "exp2",
      "query": "Qual è la capitale della Francia?",
      "should_expand": false,
      "expected_expand_node": null,
      "reason": "Completely out of scope, no node is relevant"
    }
  ],
  "relevance_checks": [
    {
      "id": "rel1",
      "node_name": "Capitolo 3 — Biologia",
      "node_context": "Questo capitolo tratta la biologia cellulare...",
      "query": "Come funziona la mitosi?",
      "expected_relevant": true
    },
    {
      "id": "rel2",
      "node_name": "Capitolo 3 — Biologia",
      "node_context": "Questo capitolo tratta la biologia cellulare...",
      "query": "Qual è il PIL dell'Italia?",
      "expected_relevant": false
    }
  ]
}
"""
import asyncio
import sys
from pathlib import Path

from config import (
    BACKEND_DIR, TEST_DATA_DIR, EvalResult, save_results, load_test_data, print_summary
)

sys.path.insert(0, str(BACKEND_DIR))

from tree_engine import TreeEngine, AUTO_EXPAND_THRESHOLD
from llm_client import check_node_relevance


async def evaluate_expansion():
    """Run expansion effectiveness evaluation."""
    print("\n🌳 EXPANSION EFFECTIVENESS EVALUATION")
    print("-" * 40)

    gt = load_test_data("expansion_ground_truth.json")
    results: list[EvalResult] = []

    # --- Part A: Auto-expansion behavior ---
    doc_path = TEST_DATA_DIR / gt["document_file"]
    if not doc_path.exists():
        print(f"  ✗ Document not found: {doc_path}")
        return []

    with open(doc_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    print("  Building tree from document...")
    engine = TreeEngine()
    await engine.build_tree_from_document(document_text, filename=gt["document_file"])
    print(f"  Tree built: {len(engine.root.children)} nodes")
    print(f"  Nodes: {[c.name for c in engine.root.children]}")

    expansion_tests = gt.get("test_cases", [])
    print(f"\n  --- Auto-Expansion Tests ({len(expansion_tests)} cases) ---")

    for tc in expansion_tests:
        test_id = tc["id"]
        query = tc["query"]
        should_expand = tc["should_expand"]
        expected_node = tc.get("expected_expand_node")

        print(f"\n  Testing: [{test_id}] {query[:50]}...")
        print(f"    Expected: should_expand={should_expand}")

        # We need a fresh engine for each test (expansion is irreversible)
        test_engine = TreeEngine()
        await test_engine.build_tree_from_document(document_text, filename=gt["document_file"])

        # Score nodes
        scores, reasons = await test_engine.score_nodes(query)
        max_score = max(scores.values()) if scores else 0.0

        # Check auto-expansion
        expanded, expanded_node_id = await test_engine.maybe_auto_expand(query, scores)

        # Determine correctness
        actually_expanded = expanded and expanded_node_id is not None

        if should_expand:
            # Should have expanded
            correct_decision = actually_expanded
            # Check if it expanded the right node
            if actually_expanded and expected_node:
                expanded_node = test_engine.get_node(expanded_node_id)
                correct_node = (
                    expected_node.lower() in expanded_node.name.lower()
                    if expanded_node else False
                )
            else:
                correct_node = not expected_node  # If no expected node, any is fine
        else:
            # Should NOT have expanded
            correct_decision = not actually_expanded
            correct_node = True  # N/A

        # Count leaves if expanded
        leaves_count = 0
        if actually_expanded:
            expanded_node_obj = test_engine.get_node(expanded_node_id)
            leaves_count = len(expanded_node_obj.children) if expanded_node_obj else 0

        metrics = {
            "correct_decision": correct_decision,
            "correct_node": correct_node,
            "actually_expanded": actually_expanded,
            "should_expand": should_expand,
            "max_score_before": round(max_score, 3),
            "expanded_node_id": expanded_node_id,
            "leaves_created": leaves_count,
        }

        passed = correct_decision and correct_node

        result = EvalResult(
            test_id=test_id,
            dimension="expansion",
            metrics=metrics,
            details={
                "query": query,
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "reason": tc.get("reason", ""),
            },
            passed=passed,
        )
        results.append(result)

        status = "✓" if passed else "✗"
        print(f"    {status} Decision={'correct' if correct_decision else 'WRONG'} "
              f"Node={'correct' if correct_node else 'WRONG'} "
              f"MaxScore={max_score:.3f} Expanded={actually_expanded}")

    # --- Part B: Relevance Check Accuracy ---
    relevance_checks = gt.get("relevance_checks", [])
    if relevance_checks:
        print(f"\n  --- Relevance Check Tests ({len(relevance_checks)} cases) ---")

        for rc in relevance_checks:
            test_id = rc["id"]
            node_name = rc["node_name"]
            node_context = rc["node_context"]
            query = rc["query"]
            expected_relevant = rc["expected_relevant"]

            print(f"\n  Testing: [{test_id}] '{node_name}' vs '{query[:40]}...'")

            is_relevant = await check_node_relevance(node_name, node_context, query)

            correct = is_relevant == expected_relevant

            metrics = {
                "correct": correct,
                "predicted_relevant": is_relevant,
                "expected_relevant": expected_relevant,
            }

            result = EvalResult(
                test_id=test_id,
                dimension="expansion_relevance",
                metrics=metrics,
                details={"node_name": node_name, "query": query},
                passed=correct,
            )
            results.append(result)

            status = "✓" if correct else "✗"
            print(f"    {status} Predicted={is_relevant} Expected={expected_relevant}")

    print_summary(results, "Expansion Effectiveness")
    save_results(results, "expansion_results.json")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
    asyncio.run(evaluate_expansion())
