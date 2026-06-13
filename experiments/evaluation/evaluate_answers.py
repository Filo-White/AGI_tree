"""
Evaluate answers using LLM judge.

Usage:
    python experiments/evaluation/evaluate_answers.py --run-dir experiments/evaluation/outputs/runs/<RUN_ID>
"""
import argparse
import asyncio
import json
import csv
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

import yaml
from metrics.answer_quality import judge_answer, compute_answer_quality_summary


def load_queries_map() -> dict:
    """Load queries.jsonl as query_id -> query_info mapping."""
    queries_file = EVAL_DIR / "data" / "queries.jsonl"
    qmap = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                q = json.loads(line)
                qmap[q["query_id"]] = q
    return qmap


def load_config(run_dir: Path) -> dict:
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
            return m.get("config", {})
    default = EVAL_DIR / "config.yaml"
    if default.exists():
        with open(default) as f:
            return yaml.safe_load(f)
    return {}


async def evaluate_run(run_dir: Path):
    """Evaluate all method results in a run directory."""
    queries_map = load_queries_map()
    config = load_config(run_dir)
    judge_model = config.get("models", {}).get("judge_model", "gpt-5.4-nano-2026-03-17")
    anonymize = config.get("evaluation", {}).get("anonymize_for_judge", True)
    sample_ratio = config.get("evaluation", {}).get("manual_validation_sample_ratio", 0.25)

    results_dir = EVAL_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_judge_results = []
    validation_samples = []

    method_dirs = [d for d in run_dir.iterdir() if d.is_dir()]

    for method_dir in sorted(method_dirs):
        method = method_dir.name
        print(f"\nJudging method: {method}")

        result_files = sorted(method_dir.glob("doc*.json"))
        judge_out_dir = method_dir / "judge"
        judge_out_dir.mkdir(exist_ok=True)

        method_results = []
        for rf in result_files:
            with open(rf, "r", encoding="utf-8") as f:
                result = json.load(f)

            query_id = result.get("query_id", rf.stem)
            query_info = queries_map.get(query_id, {})

            if not query_info:
                print(f"  SKIP {query_id}: not in queries.jsonl")
                continue

            # Build evidence text
            evidence_text = "\n\n".join(
                e.get("text", "")[:500]
                for e in result.get("selected_evidence", [])
            )

            judge_result = await judge_answer(
                query=query_info["query"],
                reference_answer=query_info.get("reference_answer", ""),
                expected_points=query_info.get("expected_answer_points", []),
                system_answer=result.get("answer", ""),
                evidence_text=evidence_text,
                answerable=query_info.get("answerable", True),
                judge_model=judge_model,
                anonymize=anonymize,
                method_name=method,
            )

            judge_result["query_id"] = query_id
            judge_result["method"] = method
            judge_result["doc_id"] = result.get("doc_id", "")

            # Save individual judge output
            judge_file = judge_out_dir / f"{query_id}_judge.json"
            with open(judge_file, "w", encoding="utf-8") as f:
                json.dump(judge_result, f, indent=2, ensure_ascii=False)

            method_results.append(judge_result)
            all_judge_results.append(judge_result)

            # Collect for manual validation
            validation_samples.append({
                "query_id": query_id,
                "doc_id": result.get("doc_id", ""),
                "method": method,
                "query": query_info["query"],
                "reference_answer": query_info.get("reference_answer", ""),
                "system_answer": result.get("answer", "")[:500],
                "evidence": evidence_text[:300],
                "judge_correctness": judge_result.get("correctness"),
                "judge_faithfulness": judge_result.get("faithfulness"),
                "judge_completeness": judge_result.get("completeness"),
                "judge_relevance": judge_result.get("relevance"),
                "human_correctness": "",
                "human_faithfulness": "",
                "human_notes": "",
            })

        # Method summary
        summary = compute_answer_quality_summary(method_results)
        summary["method"] = method
        print(f"  {method}: correctness={summary.get('correctness_mean', 'N/A')}, "
              f"faithfulness={summary.get('faithfulness_mean', 'N/A')}, "
              f"completeness={summary.get('completeness_mean', 'N/A')}")

    # Save all judge results
    judge_all_file = results_dir / "answer_quality_per_query.csv"
    if all_judge_results:
        fieldnames = ["query_id", "doc_id", "method", "correctness", "faithfulness",
                      "completeness", "relevance", "abstention_correct", "short_rationale"]
        with open(judge_all_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_judge_results)
        print(f"\nSaved per-query results to {judge_all_file}")

    # Save manual validation sample
    if validation_samples:
        random.seed(42)
        sample_size = max(1, int(len(validation_samples) * sample_ratio))
        sample = random.sample(validation_samples, min(sample_size, len(validation_samples)))
        sample_file = results_dir / "manual_validation_sample.csv"
        fieldnames = list(sample[0].keys())
        with open(sample_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sample)
        print(f"Saved validation sample ({len(sample)} items) to {sample_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    asyncio.run(evaluate_run(Path(args.run_dir)))


if __name__ == "__main__":
    main()
