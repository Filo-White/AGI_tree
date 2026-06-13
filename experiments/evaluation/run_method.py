"""
Run a single method on the evaluation dataset.

Usage:
    python experiments/evaluation/run_method.py --method dat_full --config experiments/evaluation/config.yaml
    python experiments/evaluation/run_method.py --method flat_rag --max-queries 5
"""
import argparse
import asyncio
import json
import time
import sys
import os
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


METHOD_RUNNERS = {
    "dat_full": "dat_wrappers.dat_full:run_dat_full",
    "dat_no_expansion": "dat_wrappers.dat_no_expansion:run_dat_no_expansion",
    "dat_parent_only": "dat_wrappers.dat_parent_only:run_dat_parent_only",
    "dat_no_self_scoring": "dat_wrappers.dat_no_self_scoring:run_dat_no_self_scoring",
    "flat_rag": "baselines.flat_rag:run_flat_rag",
    "section_rag": "baselines.section_rag:run_section_rag",
    "long_context": "baselines.long_context:run_long_context",
}


def load_runner(method: str):
    """Dynamically import the runner function for a method."""
    if method not in METHOD_RUNNERS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_RUNNERS.keys())}")
    module_path, func_name = METHOD_RUNNERS[method].rsplit(":", 1)
    # Import relative to EVAL_DIR
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


def load_config(config_path: str = None) -> dict:
    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    default = EVAL_DIR / "config.yaml"
    if default.exists():
        with open(default, "r") as f:
            return yaml.safe_load(f)
    return {}


def load_documents() -> dict:
    """Load documents.json and return doc_id -> doc_info mapping."""
    doc_file = EVAL_DIR / "data" / "documents.json"
    with open(doc_file, "r", encoding="utf-8") as f:
        docs = json.load(f)
    return {d["doc_id"]: d for d in docs}


def load_queries() -> list[dict]:
    """Load queries.jsonl."""
    queries_file = EVAL_DIR / "data" / "queries.jsonl"
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_document_text(doc_info: dict) -> str:
    """Load document text from path."""
    doc_path = ROOT_DIR / doc_info["path"]
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")
    if doc_path.suffix == ".pdf":
        sys.path.insert(0, str(BACKEND_DIR))
        from document_processor import process_document
        with open(doc_path, "rb") as f:
            return process_document(doc_path.name, f.read())
    else:
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


async def run_method(
    method: str,
    config: dict,
    max_queries: int = None,
    run_id: str = None,
    doc_filter: str = None,
):
    """Run a method on the full dataset."""
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    runner = load_runner(method)
    documents = load_documents()
    queries = load_queries()

    if doc_filter:
        queries = [q for q in queries if q["doc_id"] == doc_filter]
    if max_queries:
        queries = queries[:max_queries]

    # Output directory
    out_dir = EVAL_DIR / "outputs" / "runs" / run_id / method
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {method} on {len(queries)} queries (run_id={run_id})")

    results = []
    doc_cache = {}

    for i, q in enumerate(queries):
        doc_id = q["doc_id"]
        query_id = q["query_id"]

        # Load document (cached)
        if doc_id not in doc_cache:
            doc_info = documents.get(doc_id)
            if not doc_info:
                print(f"  SKIP {query_id}: doc {doc_id} not in documents.json")
                continue
            try:
                doc_cache[doc_id] = load_document_text(doc_info)
            except FileNotFoundError as e:
                print(f"  SKIP {query_id}: {e}")
                continue

        doc_text = doc_cache[doc_id]
        print(f"  [{i+1}/{len(queries)}] {query_id}: {q['query'][:60]}...")

        try:
            result = await runner(doc_text, q["query"], config, doc_id=doc_id)
            result["run_id"] = run_id
            result["timestamp"] = datetime.now().isoformat()
            result["query_id"] = query_id
        except Exception as e:
            result = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "query_id": query_id,
                "doc_id": doc_id,
                "query": q["query"],
                "answer": "",
                "selected_evidence": [],
                "dat_trace": None,
                "usage": {},
                "errors": [str(e)],
            }
            print(f"    ERROR: {e}")

        # Save per-query result
        result_file = out_dir / f"{query_id}.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        results.append(result)

    # Save summary
    summary_file = out_dir / "_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "method": method,
            "run_id": run_id,
            "n_queries": len(results),
            "n_errors": sum(1 for r in results if r.get("errors")),
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"  Done. {len(results)} results saved to {out_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run a single evaluation method")
    parser.add_argument("--method", required=True, choices=list(METHOD_RUNNERS.keys()))
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--doc", default=None, help="Filter by doc_id")
    args = parser.parse_args()

    config = load_config(args.config)
    asyncio.run(run_method(args.method, config, args.max_queries, args.run_id, args.doc))


if __name__ == "__main__":
    main()
