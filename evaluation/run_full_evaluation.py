"""
AGI Tree — Full Multi-Document Evaluation
==========================================
Runs all 4 evaluation dimensions on all 3 test documents and produces a combined report.

Documents:
  1. test_doc.txt (Book — Intelligenza Artificiale)
  2. test_paper.txt (Paper — Reinforcement Learning in Robotics)
  3. test_administrative.txt (Administrative — Regolamento Campus)
"""
import asyncio
import sys
import json
import time
from pathlib import Path

from config import (
    BACKEND_DIR, TEST_DATA_DIR, RESULTS_DIR, EvalResult, save_results, print_summary
)

sys.path.insert(0, str(BACKEND_DIR))


# ============================================================
# SCORING EVALUATION (all docs)
# ============================================================
async def run_scoring_all():
    """Run scoring evaluation on all 3 documents."""
    from eval_scoring import evaluate_scoring
    from config import load_test_data

    gt_files = [
        "scoring_ground_truth.json",  # book
        "scoring_gt_paper.json",      # paper
        "scoring_gt_admin.json",      # administrative
    ]

    all_results = []
    for gt_file in gt_files:
        print(f"\n{'─'*60}")
        print(f"  Scoring: {gt_file}")
        print(f"{'─'*60}")
        try:
            # Temporarily swap the ground truth
            gt = load_test_data(gt_file)
            doc_path = TEST_DATA_DIR / gt["document_file"]
            if not doc_path.exists():
                print(f"  ⚠ Document not found: {doc_path}")
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                document_text = f.read()

            from tree_engine import TreeEngine
            from llm_client import get_competence_score

            engine = TreeEngine()
            await engine.build_tree_from_document(document_text, filename=gt["document_file"])
            print(f"  Tree: {len(engine.root.children)} nodes")

            node_names = [c.name for c in engine.root.children]

            for tc in gt["test_cases"]:
                test_id = tc["id"]
                query = tc["query"]
                correct_names = [n.lower() for n in tc["correct_node_names"]]
                irrelevant_names = [n.lower() for n in tc.get("irrelevant_node_names", [])]

                scores, reasons = await engine.score_nodes(query)

                name_scores = {}
                for node_id, score in scores.items():
                    node = engine.get_node(node_id)
                    if node:
                        name_scores[node.name] = score

                ranked = sorted(name_scores.items(), key=lambda x: x[1], reverse=True)

                top1_name = ranked[0][0].lower() if ranked else ""
                hit_at_1 = any(c in top1_name for c in correct_names)

                top3_names = [r[0].lower() for r in ranked[:3]]
                hit_at_3 = any(any(c in t3 for c in correct_names) for t3 in top3_names)

                correct_scores = [s for name, s in name_scores.items() if any(c in name.lower() for c in correct_names)]
                irrelevant_scores = [s for name, s in name_scores.items() if any(c in name.lower() for c in irrelevant_names)]
                avg_correct = sum(correct_scores) / len(correct_scores) if correct_scores else 0
                avg_irrelevant = sum(irrelevant_scores) / len(irrelevant_scores) if irrelevant_scores else 0
                score_separation = avg_correct - avg_irrelevant

                passed = hit_at_3 and score_separation > 0

                result = EvalResult(
                    test_id=test_id,
                    dimension="scoring",
                    metrics={
                        "hit@1": hit_at_1,
                        "hit@3": hit_at_3,
                        "score_separation": round(score_separation, 3),
                        "avg_correct_score": round(avg_correct, 3),
                        "avg_irrelevant_score": round(avg_irrelevant, 3),
                    },
                    details={"query": query, "doc": gt["document_file"]},
                    passed=passed,
                )
                all_results.append(result)
                status = "✓" if passed else "✗"
                print(f"  {status} [{test_id}] Hit@1={hit_at_1} Hit@3={hit_at_3} Sep={score_separation:.3f}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return all_results


# ============================================================
# SPLITTING EVALUATION (all docs)
# ============================================================
async def run_splitting_all():
    """Run splitting evaluation on all 3 documents."""
    from eval_splitting import evaluate_splitting, fuzzy_match

    gt_files = [
        "splitting_ground_truth.json",
        "splitting_gt_paper.json",
        "splitting_gt_admin.json",
    ]

    all_results = []
    for gt_file in gt_files:
        print(f"\n{'─'*60}")
        print(f"  Splitting: {gt_file}")
        print(f"{'─'*60}")
        try:
            from config import load_test_data
            from tree_engine import TreeEngine
            from llm_client import classify_document

            gt = load_test_data(gt_file)
            for tc in gt["test_cases"]:
                test_id = tc["id"]
                doc_path = TEST_DATA_DIR / tc["document_file"]
                if not doc_path.exists():
                    print(f"  ⚠ [{test_id}] Document not found")
                    continue

                with open(doc_path, "r", encoding="utf-8") as f:
                    document_text = f.read()

                classification = await classify_document(document_text)
                detected_type = classification.get("doc_type", "unknown")
                expected_type = tc["expected_doc_type"]
                type_correct = detected_type == expected_type

                engine = TreeEngine()
                await engine.build_tree_from_document(document_text, filename=tc["document_file"])

                actual_names = [n.name for n in engine.root.children]
                expected_sections = tc["expected_sections"]
                min_nodes = tc.get("min_nodes", 2)
                max_nodes = tc.get("max_nodes", 20)

                count_in_range = min_nodes <= len(actual_names) <= max_nodes

                matched = 0
                for exp_name in expected_sections:
                    if fuzzy_match(exp_name, actual_names):
                        matched += 1
                name_coverage = matched / len(expected_sections) if expected_sections else 0

                covered_chars = sum(len(nd.get("text", "")) for nd in engine._nodes_data)
                text_coverage = min(covered_chars / len(document_text), 1.0)

                passed = type_correct and count_in_range and name_coverage >= 0.6 and text_coverage >= 0.8

                result = EvalResult(
                    test_id=test_id,
                    dimension="splitting",
                    metrics={
                        "classification_correct": type_correct,
                        "detected_type": detected_type,
                        "expected_type": expected_type,
                        "node_count": len(actual_names),
                        "count_in_range": count_in_range,
                        "name_coverage": round(name_coverage, 3),
                        "text_coverage": round(text_coverage, 3),
                    },
                    details={"actual_names": actual_names, "doc": tc["document_file"]},
                    passed=passed,
                )
                all_results.append(result)
                status = "✓" if passed else "✗"
                print(f"  {status} [{test_id}] Type={detected_type}({'✓' if type_correct else '✗'}) "
                      f"Nodes={len(actual_names)} NameCov={name_coverage:.0%} TextCov={text_coverage:.0%}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return all_results


# ============================================================
# EXPANSION EVALUATION (all docs)
# ============================================================
async def run_expansion_all():
    """Run expansion evaluation on all 3 documents."""
    from config import load_test_data
    from tree_engine import TreeEngine
    from llm_client import check_node_relevance

    gt_files = [
        "expansion_ground_truth.json",
        "expansion_gt_paper.json",
        "expansion_gt_admin.json",
    ]

    all_results = []
    for gt_file in gt_files:
        print(f"\n{'─'*60}")
        print(f"  Expansion: {gt_file}")
        print(f"{'─'*60}")
        try:
            gt = load_test_data(gt_file)
            doc_path = TEST_DATA_DIR / gt["document_file"]
            if not doc_path.exists():
                print(f"  ⚠ Document not found: {doc_path}")
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                document_text = f.read()

            # Auto-expansion tests
            for tc in gt.get("test_cases", []):
                test_id = tc["id"]
                query = tc["query"]
                should_expand = tc["should_expand"]

                test_engine = TreeEngine()
                await test_engine.build_tree_from_document(document_text, filename=gt["document_file"])

                scores, reasons = await test_engine.score_nodes(query)
                max_score = max(scores.values()) if scores else 0.0
                expanded, expanded_node_id = await test_engine.maybe_auto_expand(query, scores)
                actually_expanded = expanded and expanded_node_id is not None

                correct_decision = actually_expanded == should_expand

                result = EvalResult(
                    test_id=test_id,
                    dimension="expansion",
                    metrics={
                        "correct_decision": correct_decision,
                        "should_expand": should_expand,
                        "actually_expanded": actually_expanded,
                        "max_score": round(max_score, 3),
                    },
                    details={"query": query, "doc": gt["document_file"]},
                    passed=correct_decision,
                )
                all_results.append(result)
                status = "✓" if correct_decision else "✗"
                print(f"  {status} [{test_id}] Decision={'correct' if correct_decision else 'WRONG'} "
                      f"MaxScore={max_score:.3f} Expanded={actually_expanded}")

            # Relevance checks
            for rc in gt.get("relevance_checks", []):
                test_id = rc["id"]
                is_relevant = await check_node_relevance(rc["node_name"], rc["node_context"], rc["query"])
                correct = is_relevant == rc["expected_relevant"]

                result = EvalResult(
                    test_id=test_id,
                    dimension="relevance_check",
                    metrics={
                        "correct": correct,
                        "predicted": is_relevant,
                        "expected": rc["expected_relevant"],
                    },
                    details={"node": rc["node_name"], "query": rc["query"], "doc": gt["document_file"]},
                    passed=correct,
                )
                all_results.append(result)
                status = "✓" if correct else "✗"
                print(f"  {status} [{test_id}] Pred={is_relevant} Exp={rc['expected_relevant']}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    return all_results


# ============================================================
# EFFICIENCY EVALUATION (all docs)
# ============================================================
async def run_efficiency_all():
    """Run efficiency evaluation on all 3 documents."""
    import time as _time
    from config import load_test_data
    from tree_engine import TreeEngine
    from eval_efficiency import _tracker, _patch_client, _measure_start, _measure_end

    gt_files = [
        "efficiency_test.json",
        "efficiency_paper.json",
        "efficiency_admin.json",
    ]

    _patch_client()

    all_results = []
    all_measurements = {}

    for gt_file in gt_files:
        print(f"\n{'─'*60}")
        print(f"  Efficiency: {gt_file}")
        print(f"{'─'*60}")
        try:
            gt = load_test_data(gt_file)
            doc_path = TEST_DATA_DIR / gt["document_file"]
            if not doc_path.exists():
                print(f"  ⚠ Document not found")
                continue

            with open(doc_path, "r", encoding="utf-8") as f:
                document_text = f.read()

            doc_key = gt["document_file"]

            # Build tree
            engine = TreeEngine()
            start = _measure_start()
            await engine.build_tree_from_document(document_text, filename=doc_key)
            build_m = _measure_end(start)
            print(f"  Build: {build_m['total_calls']} calls, {build_m['total_tokens']} tokens, "
                  f"{build_m['wall_time_ms']:.0f}ms")

            # Scoring
            queries = gt.get("queries", [])
            score_measurements = []
            for q in queries:
                start = _measure_start()
                await engine.score_nodes(q)
                sm = _measure_end(start)
                score_measurements.append(sm)

            avg_score_tokens = sum(m["total_tokens"] for m in score_measurements) / len(score_measurements) if score_measurements else 0
            avg_score_time = sum(m["wall_time_ms"] for m in score_measurements) / len(score_measurements) if score_measurements else 0
            avg_score_calls = sum(m["total_calls"] for m in score_measurements) / len(score_measurements) if score_measurements else 0
            print(f"  Score (avg): {avg_score_calls:.0f} calls, {avg_score_tokens:.0f} tokens, "
                  f"{avg_score_time:.0f}ms")

            # Expansion
            expand_name = gt.get("expand_node_name")
            expand_m = None
            if expand_name:
                target = None
                for child in engine.root.children:
                    if expand_name.lower() in child.name.lower() and not child.children:
                        target = child
                        break
                if target:
                    start = _measure_start()
                    await engine.expand_node(target.id)
                    expand_m = _measure_end(start)
                    print(f"  Expand: {expand_m['total_calls']} calls, {expand_m['total_tokens']} tokens, "
                          f"{expand_m['wall_time_ms']:.0f}ms, {len(target.children)} leaves")

            # Full query
            engine2 = TreeEngine()
            _tracker.reset()
            await engine2.build_tree_from_document(document_text, filename=doc_key)
            query_measurements = []
            for q in queries[:3]:
                start = _measure_start()
                await engine2.process_query(q)
                qm = _measure_end(start)
                query_measurements.append(qm)

            avg_q_tokens = sum(m["total_tokens"] for m in query_measurements) / len(query_measurements) if query_measurements else 0
            avg_q_time = sum(m["wall_time_ms"] for m in query_measurements) / len(query_measurements) if query_measurements else 0
            avg_q_calls = sum(m["total_calls"] for m in query_measurements) / len(query_measurements) if query_measurements else 0
            print(f"  Query (avg): {avg_q_calls:.0f} calls, {avg_q_tokens:.0f} tokens, "
                  f"{avg_q_time:.0f}ms")

            doc_metrics = {
                "document": doc_key,
                "build_calls": build_m["total_calls"],
                "build_tokens": build_m["total_tokens"],
                "build_time_ms": round(build_m["wall_time_ms"]),
                "scoring_avg_calls": round(avg_score_calls, 1),
                "scoring_avg_tokens": round(avg_score_tokens),
                "scoring_avg_time_ms": round(avg_score_time),
                "expand_calls": expand_m["total_calls"] if expand_m else None,
                "expand_tokens": expand_m["total_tokens"] if expand_m else None,
                "expand_time_ms": round(expand_m["wall_time_ms"]) if expand_m else None,
                "query_avg_calls": round(avg_q_calls, 1),
                "query_avg_tokens": round(avg_q_tokens),
                "query_avg_time_ms": round(avg_q_time),
            }
            all_measurements[doc_key] = doc_metrics

            all_results.append(EvalResult(
                test_id=f"eff_{doc_key}",
                dimension="efficiency",
                metrics=doc_metrics,
                details={},
                passed=True,
            ))

        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()

    return all_results, all_measurements


# ============================================================
# MAIN
# ============================================================
async def main():
    print("=" * 70)
    print("  AGI TREE — FULL MULTI-DOCUMENT EVALUATION")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = []

    # 1. Scoring
    print("\n\n" + "█" * 70)
    print("  DIMENSION 1: SCORING QUALITY")
    print("█" * 70)
    scoring_results = await run_scoring_all()
    all_results.extend(scoring_results)

    # 2. Splitting
    print("\n\n" + "█" * 70)
    print("  DIMENSION 2: SPLITTING QUALITY")
    print("█" * 70)
    splitting_results = await run_splitting_all()
    all_results.extend(splitting_results)

    # 3. Expansion
    print("\n\n" + "█" * 70)
    print("  DIMENSION 3: EXPANSION EFFECTIVENESS")
    print("█" * 70)
    expansion_results = await run_expansion_all()
    all_results.extend(expansion_results)

    # 4. Efficiency
    print("\n\n" + "█" * 70)
    print("  DIMENSION 4: EFFICIENCY")
    print("█" * 70)
    efficiency_results, efficiency_measurements = await run_efficiency_all()
    all_results.extend(efficiency_results)

    # ============================================================
    # FINAL COMBINED REPORT
    # ============================================================
    print("\n\n" + "=" * 70)
    print("  FINAL COMBINED RESULTS")
    print("=" * 70)

    by_dimension = {}
    for r in all_results:
        dim = r.dimension
        if dim not in by_dimension:
            by_dimension[dim] = {"total": 0, "passed": 0}
        by_dimension[dim]["total"] += 1
        if r.passed:
            by_dimension[dim]["passed"] += 1

    for dim, stats in by_dimension.items():
        pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {dim:<25} {bar} {stats['passed']}/{stats['total']} ({pct:.0f}%)")

    total = len(all_results)
    total_passed = sum(1 for r in all_results if r.passed)
    print(f"\n  Overall: {total_passed}/{total} tests passed ({total_passed/total*100:.0f}%)")
    print("=" * 70)

    # Save combined results
    save_results(all_results, "full_evaluation_results.json")

    # Save efficiency summary separately
    eff_path = RESULTS_DIR / "efficiency_summary.json"
    with open(eff_path, "w", encoding="utf-8") as f:
        json.dump(efficiency_measurements, f, indent=2, ensure_ascii=False)

    return all_results, efficiency_measurements


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
    asyncio.run(main())
