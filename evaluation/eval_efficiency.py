"""
Evaluation: Efficiency
=======================
Measures computational cost of the system:
1. Token usage per operation (build tree, score, expand, full query)
2. Latency (wall-clock time) per operation
3. Number of LLM calls per operation
4. Cost breakdown by phase

Requires: test_data/efficiency_test.json
Format:
{
  "document_file": "test_doc.txt",
  "queries": [
    "Query 1 to test",
    "Query 2 to test"
  ],
  "expand_node_name": "optional - name of node to expand for testing"
}

This module monkey-patches the OpenAI client to intercept all calls and log token usage.
"""
import asyncio
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from unittest.mock import patch

from config import (
    BACKEND_DIR, TEST_DATA_DIR, RESULTS_DIR, EvalResult, save_results, load_test_data, print_summary
)

sys.path.insert(0, str(BACKEND_DIR))


@dataclass
class CallMetrics:
    """Aggregated metrics for a set of LLM calls."""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    calls: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_calls if self.total_calls > 0 else 0

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class LLMCallTracker:
    """Context manager that tracks all OpenAI API calls made during its scope."""

    def __init__(self):
        self.metrics = CallMetrics()
        self._original_create = None

    async def _tracked_create(self, *args, **kwargs):
        """Wrapper around the original create method that logs metrics."""
        start = time.perf_counter()
        response = await self._original_create(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

        self.metrics.total_calls += 1
        self.metrics.total_input_tokens += input_tokens
        self.metrics.total_output_tokens += output_tokens
        self.metrics.total_latency_ms += elapsed_ms

        self.metrics.calls.append({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(elapsed_ms, 1),
            "model": kwargs.get("model", "unknown"),
        })

        return response

    def reset(self):
        """Reset all tracked metrics."""
        self.metrics = CallMetrics()


# Global tracker instance
_tracker = LLMCallTracker()

from tree_engine import TreeEngine


def _patch_client():
    """Patch the OpenAI client to track calls. Must be called after client is initialized."""
    from llm_client import _get_client

    client = _get_client()
    if _tracker._original_create is None:
        _tracker._original_create = client.chat.completions.create
    client.chat.completions.create = _tracker._tracked_create


def _measure_start():
    """Start a new measurement window."""
    _tracker.reset()
    return time.perf_counter()


def _measure_end(start_time: float) -> dict:
    """End measurement and return results."""
    wall_time_ms = (time.perf_counter() - start_time) * 1000
    result = _tracker.metrics.to_dict()
    result["wall_time_ms"] = round(wall_time_ms, 1)
    return result


async def evaluate_efficiency():
    """Run efficiency evaluation."""
    print("\n⚡ EFFICIENCY EVALUATION")
    print("-" * 40)

    gt = load_test_data("efficiency_test.json")
    doc_path = TEST_DATA_DIR / gt["document_file"]

    if not doc_path.exists():
        print(f"  ✗ Document not found: {doc_path}")
        return []

    with open(doc_path, "r", encoding="utf-8") as f:
        document_text = f.read()

    # Initialize and patch client
    _patch_client()

    results: list[EvalResult] = []
    all_measurements = {}

    # --- Measurement 1: Tree Building ---
    print("\n  📊 Measuring: Tree Building")
    engine = TreeEngine()
    start = _measure_start()
    await engine.build_tree_from_document(document_text, filename=gt["document_file"])
    build_metrics = _measure_end(start)
    all_measurements["build_tree"] = build_metrics
    print(f"    Calls: {build_metrics['total_calls']} | "
          f"Tokens: {build_metrics['total_tokens']} | "
          f"Time: {build_metrics['wall_time_ms']:.0f}ms")

    results.append(EvalResult(
        test_id="build_tree",
        dimension="efficiency",
        metrics=build_metrics,
        details={"document_chars": len(document_text), "nodes_created": len(engine.root.children)},
        passed=True,  # No pass/fail for efficiency, just measurement
    ))

    # --- Measurement 2: Scoring (per query) ---
    queries = gt.get("queries", [])
    scoring_measurements = []

    for i, query in enumerate(queries):
        print(f"\n  📊 Measuring: Scoring query {i+1}/{len(queries)}")
        start = _measure_start()
        scores, reasons = await engine.score_nodes(query)
        score_metrics = _measure_end(start)
        scoring_measurements.append(score_metrics)
        print(f"    Calls: {score_metrics['total_calls']} | "
              f"Tokens: {score_metrics['total_tokens']} | "
              f"Time: {score_metrics['wall_time_ms']:.0f}ms")

    if scoring_measurements:
        avg_scoring = {
            "avg_calls": round(sum(m["total_calls"] for m in scoring_measurements) / len(scoring_measurements), 1),
            "avg_tokens": round(sum(m["total_tokens"] for m in scoring_measurements) / len(scoring_measurements), 0),
            "avg_input_tokens": round(sum(m["total_input_tokens"] for m in scoring_measurements) / len(scoring_measurements), 0),
            "avg_output_tokens": round(sum(m["total_output_tokens"] for m in scoring_measurements) / len(scoring_measurements), 0),
            "avg_latency_ms": round(sum(m["wall_time_ms"] for m in scoring_measurements) / len(scoring_measurements), 1),
        }
        all_measurements["scoring_avg"] = avg_scoring
        results.append(EvalResult(
            test_id="scoring_avg",
            dimension="efficiency",
            metrics=avg_scoring,
            details={"num_queries": len(queries), "num_nodes": len(engine.root.children)},
            passed=True,
        ))

    # --- Measurement 3: Node Expansion ---
    expand_node_name = gt.get("expand_node_name")
    if expand_node_name:
        target_node = None
        for child in engine.root.children:
            if expand_node_name.lower() in child.name.lower():
                target_node = child
                break

        if target_node and not target_node.children:
            print(f"\n  📊 Measuring: Node Expansion ({target_node.name})")
            start = _measure_start()
            await engine.expand_node(target_node.id)
            expand_metrics = _measure_end(start)
            all_measurements["expand_node"] = expand_metrics
            leaves_created = len(target_node.children)
            print(f"    Calls: {expand_metrics['total_calls']} | "
                  f"Tokens: {expand_metrics['total_tokens']} | "
                  f"Time: {expand_metrics['wall_time_ms']:.0f}ms | "
                  f"Leaves: {leaves_created}")

            results.append(EvalResult(
                test_id="expand_node",
                dimension="efficiency",
                metrics=expand_metrics,
                details={"node_name": target_node.name, "leaves_created": leaves_created},
                passed=True,
            ))

    # --- Measurement 4: Full Query Pipeline ---
    if queries:
        print(f"\n  📊 Measuring: Full Query Pipeline")
        # Need fresh engine with no expansion
        engine2 = TreeEngine()
        _tracker.reset()  # Don't count rebuild
        await engine2.build_tree_from_document(document_text, filename=gt["document_file"])

        query_measurements = []
        for i, query in enumerate(queries[:3]):  # Limit to 3 for cost
            print(f"    Query {i+1}: {query[:50]}...")
            start = _measure_start()
            result = await engine2.process_query(query)
            q_metrics = _measure_end(start)
            q_metrics["auto_expanded"] = result.get("auto_expanded") is not None
            query_measurements.append(q_metrics)
            print(f"      Calls: {q_metrics['total_calls']} | "
                  f"Tokens: {q_metrics['total_tokens']} | "
                  f"Time: {q_metrics['wall_time_ms']:.0f}ms | "
                  f"AutoExpand: {q_metrics['auto_expanded']}")

        if query_measurements:
            avg_query = {
                "avg_calls": round(sum(m["total_calls"] for m in query_measurements) / len(query_measurements), 1),
                "avg_tokens": round(sum(m["total_tokens"] for m in query_measurements) / len(query_measurements), 0),
                "avg_latency_ms": round(sum(m["wall_time_ms"] for m in query_measurements) / len(query_measurements), 1),
                "max_calls": max(m["total_calls"] for m in query_measurements),
                "max_tokens": max(m["total_tokens"] for m in query_measurements),
            }
            all_measurements["full_query_avg"] = avg_query
            results.append(EvalResult(
                test_id="full_query_avg",
                dimension="efficiency",
                metrics=avg_query,
                details={"num_queries_tested": len(query_measurements)},
                passed=True,
            ))

    # --- Summary Table ---
    print(f"\n{'='*70}")
    print(f"  EFFICIENCY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Operation':<20} {'Calls':<8} {'Tokens':<10} {'Time (ms)':<12}")
    print(f"  {'-'*50}")
    if "build_tree" in all_measurements:
        m = all_measurements["build_tree"]
        print(f"  {'Build Tree':<20} {m['total_calls']:<8} {m['total_tokens']:<10} {m['wall_time_ms']:<12.0f}")
    if "scoring_avg" in all_measurements:
        m = all_measurements["scoring_avg"]
        print(f"  {'Scoring (avg)':<20} {m['avg_calls']:<8} {int(m['avg_tokens']):<10} {m['avg_latency_ms']:<12.0f}")
    if "expand_node" in all_measurements:
        m = all_measurements["expand_node"]
        print(f"  {'Expand Node':<20} {m['total_calls']:<8} {m['total_tokens']:<10} {m['wall_time_ms']:<12.0f}")
    if "full_query_avg" in all_measurements:
        m = all_measurements["full_query_avg"]
        print(f"  {'Full Query (avg)':<20} {m['avg_calls']:<8} {int(m['avg_tokens']):<10} {m['avg_latency_ms']:<12.0f}")
    print(f"{'='*70}\n")

    save_results(results, "efficiency_results.json")
    return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")
    asyncio.run(evaluate_efficiency())
