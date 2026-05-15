"""
Shared configuration and utilities for AGI Tree evaluation.
"""
import sys
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Add backend to path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Paths
EVAL_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = EVAL_DIR / "test_data"
RESULTS_DIR = EVAL_DIR / "results"

# Ensure directories exist
TEST_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class LLMCallLog:
    """Logs a single LLM API call."""
    function: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvalResult:
    """Container for evaluation results of a single test case."""
    test_id: str
    dimension: str  # scoring | splitting | expansion | efficiency
    metrics: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def save_results(results: list[EvalResult], filename: str):
    """Save evaluation results to JSON file."""
    out_path = RESULTS_DIR / filename
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
        "results": [r.to_dict() for r in results],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {out_path}")
    return data


def load_test_data(filename: str) -> dict:
    """Load a ground truth / test data JSON file."""
    path = TEST_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Test data file not found: {path}\n"
            f"Please create it following the format in evaluation/README.md"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_summary(results: list[EvalResult], dimension: str):
    """Print a formatted summary of results."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print(f"\n{'='*60}")
    print(f"  {dimension.upper()} EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total tests: {total}")
    print(f"  Passed:      {passed} ({passed/total*100:.1f}%)" if total > 0 else "  Passed: 0")
    print(f"  Failed:      {total - passed}")
    print(f"{'='*60}")

    for r in results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} [{r.test_id}]")
        for k, v in r.metrics.items():
            print(f"      {k}: {v}")
    print()
