"""
Run all evaluation methods on the dataset.

Usage:
    python experiments/evaluation/run_all.py --dry-run --max-queries 3
    python experiments/evaluation/run_all.py --config experiments/evaluation/config.yaml
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

import yaml

EVAL_DIR = Path(__file__).resolve().parent
ROOT_DIR = EVAL_DIR.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(EVAL_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

ALL_METHODS = [
    "dat_full",
    "flat_rag",
    "section_rag",
    "long_context",
    "dat_no_expansion",
    "dat_parent_only",
    "dat_no_self_scoring",
]

MAIN_METHODS = [
    "dat_full",
    "flat_rag",
    "section_rag",
    "long_context",
]


def load_config(config_path: str = None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    default = EVAL_DIR / "config.yaml"
    if default.exists():
        with open(default, "r") as f:
            return yaml.safe_load(f)
    return {}


async def main():
    parser = argparse.ArgumentParser(description="Run all evaluation methods")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Run on max 3 queries only")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--methods", nargs="+", default=None, help="Methods to run (default: all)")
    parser.add_argument("--main-only", action="store_true", help="Run only main methods (no ablations)")
    parser.add_argument("--doc", default=None, help="Filter by doc_id")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    max_queries = args.max_queries
    if args.dry_run:
        max_queries = max_queries or 3
        print(f"=== DRY RUN (max {max_queries} queries) ===\n")

    methods = args.methods or (MAIN_METHODS if args.main_only else ALL_METHODS)

    print(f"Run ID: {run_id}")
    print(f"Methods: {methods}")
    print(f"Max queries: {max_queries or 'all'}")
    print()

    from run_method import run_method

    all_results = {}
    for method in methods:
        print(f"\n{'='*60}")
        print(f"  METHOD: {method}")
        print(f"{'='*60}")
        try:
            results = await run_method(method, config, max_queries, run_id, args.doc)
            all_results[method] = results
        except Exception as e:
            print(f"  FAILED: {e}")
            all_results[method] = []

    # Save run manifest
    manifest_dir = EVAL_DIR / "outputs" / "runs" / run_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "methods": methods,
        "max_queries": max_queries,
        "dry_run": args.dry_run,
        "config": config,
        "results_summary": {
            m: {"n_queries": len(r), "n_errors": sum(1 for x in r if x.get("errors"))}
            for m, r in all_results.items()
        },
    }
    with open(manifest_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"  ALL METHODS COMPLETE")
    print(f"  Run ID: {run_id}")
    print(f"  Output: {manifest_dir}")
    for m, r in all_results.items():
        errors = sum(1 for x in r if x.get("errors"))
        print(f"  {m}: {len(r)} queries, {errors} errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
