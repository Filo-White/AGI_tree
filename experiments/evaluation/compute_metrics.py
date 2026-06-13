"""
Compute all aggregate metrics from run outputs.

Usage:
    python experiments/evaluation/compute_metrics.py --run-dir experiments/evaluation/outputs/runs/<RUN_ID>
"""
import argparse
import csv
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent.parent
sys.path.insert(0, str(EVAL_DIR))

from metrics.routing import compute_routing_metrics, compute_dat_score_separation, compute_routing_summary
from metrics.expansion import compute_expansion_metrics, compute_expansion_summary
from metrics.efficiency import compute_efficiency_metrics, compute_efficiency_summary
from metrics.answer_quality import compute_answer_quality_summary


def load_queries_map() -> dict:
    queries_file = EVAL_DIR / "data" / "queries.jsonl"
    qmap = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                qmap[q["query_id"]] = q
    return qmap


def load_run_results(run_dir: Path) -> dict:
    """Load all per-query results grouped by method."""
    results = {}
    for method_dir in sorted(run_dir.iterdir()):
        if not method_dir.is_dir():
            continue
        method = method_dir.name
        method_results = []
        for rf in sorted(method_dir.glob("doc*.json")):
            with open(rf, "r", encoding="utf-8") as f:
                method_results.append(json.load(f))
        if method_results:
            results[method] = method_results
    return results


def compute_all_metrics(run_dir: Path):
    queries_map = load_queries_map()
    all_results = load_run_results(run_dir)
    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Answer Quality Summary ---
    quality_summary_rows = []
    judge_dir_check = False
    for method, results in all_results.items():
        method_dir = run_dir / method / "judge"
        if method_dir.exists():
            judge_dir_check = True
            judge_results = []
            for jf in sorted(method_dir.glob("*_judge.json")):
                with open(jf) as f:
                    judge_results.append(json.load(f))
            if judge_results:
                summary = compute_answer_quality_summary(judge_results)
                summary["method"] = method
                quality_summary_rows.append(summary)

    if quality_summary_rows:
        _write_csv(results_dir / "answer_quality_summary.csv", quality_summary_rows)
        print(f"Saved answer_quality_summary.csv")

    # --- Routing Metrics ---
    routing_rows = []
    for method, results in all_results.items():
        for r in results:
            qid = r.get("query_id", "")
            qinfo = queries_map.get(qid, {})
            supporting = qinfo.get("supporting_sections", [])
            evidence = r.get("selected_evidence", [])
            rm = compute_routing_metrics(evidence, supporting)
            rm["query_id"] = qid
            rm["method"] = method
            rm["doc_id"] = r.get("doc_id", "")

            # DAT-specific
            dat_trace = r.get("dat_trace")
            if dat_trace and dat_trace.get("node_scores"):
                ds = compute_dat_score_separation(dat_trace["node_scores"], supporting)
                rm.update(ds)

            routing_rows.append(rm)

    if routing_rows:
        _write_csv(results_dir / "routing_metrics.csv", routing_rows)
        print(f"Saved routing_metrics.csv")

    # --- Expansion Metrics ---
    expansion_rows = []
    for method, results in all_results.items():
        for r in results:
            qid = r.get("query_id", "")
            qinfo = queries_map.get(qid, {})
            dat_trace = r.get("dat_trace")
            if dat_trace is not None:
                em = compute_expansion_metrics(
                    expansion_triggered=dat_trace.get("expansion_triggered", False),
                    requires_expansion=qinfo.get("requires_expansion", False),
                    expanded_node=dat_trace.get("expanded_node"),
                    supporting_sections=qinfo.get("supporting_sections", []),
                )
                em["query_id"] = qid
                em["method"] = method
                expansion_rows.append(em)

    if expansion_rows:
        _write_csv(results_dir / "expansion_metrics.csv", expansion_rows)
        print(f"Saved expansion_metrics.csv")

    # --- Efficiency Metrics ---
    eff_rows = []
    eff_summaries = []
    for method, results in all_results.items():
        method_eff = []
        for r in results:
            usage = r.get("usage", {})
            em = compute_efficiency_metrics(usage)
            em["query_id"] = r.get("query_id", "")
            em["method"] = method
            em["doc_id"] = r.get("doc_id", "")
            eff_rows.append(em)
            method_eff.append(em)
        if method_eff:
            eff_summaries.append(compute_efficiency_summary(method_eff, method))

    if eff_rows:
        _write_csv(results_dir / "efficiency_per_query.csv", eff_rows)
        print(f"Saved efficiency_per_query.csv")

    if eff_summaries:
        _write_csv(results_dir / "efficiency_summary.csv", eff_summaries)
        print(f"Saved efficiency_summary.csv")

    # --- Ablation Summary ---
    ablation_methods = ["dat_full", "dat_no_expansion", "dat_parent_only", "dat_no_self_scoring"]
    ablation_rows = []
    for method in ablation_methods:
        row = {"method": method}
        # Quality
        for qs in quality_summary_rows:
            if qs.get("method") == method:
                row["correctness"] = qs.get("correctness_mean")
                row["faithfulness"] = qs.get("faithfulness_mean")
                break
        # Routing
        method_routing = [r for r in routing_rows if r.get("method") == method]
        if method_routing:
            rs = compute_routing_summary(method_routing)
            row["hit_at_3"] = rs.get("hit_at_3_mean")
        # Efficiency
        for es in eff_summaries:
            if es.get("method") == method:
                row["avg_total_tokens"] = es.get("avg_total_tokens")
                row["avg_latency"] = es.get("avg_latency_seconds")
                break
        ablation_rows.append(row)

    if ablation_rows:
        _write_csv(results_dir / "ablation_summary.csv", ablation_rows)
        print(f"Saved ablation_summary.csv")

    print("\nAll metrics computed.")


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    # Collect all keys
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    compute_all_metrics(Path(args.run_dir))


if __name__ == "__main__":
    main()
