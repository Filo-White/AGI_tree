"""
AGI Tree Evaluation — Run All Dimensions
==========================================
Runs all evaluation scripts and produces a combined report.

Usage:
    cd evaluation
    python run_all.py              # Run all evaluations
    python run_all.py scoring      # Run only scoring
    python run_all.py splitting    # Run only splitting
    python run_all.py expansion    # Run only expansion
    python run_all.py efficiency   # Run only efficiency
"""
import asyncio
import sys
import json
import time
from pathlib import Path

from config import BACKEND_DIR, RESULTS_DIR, save_results

sys.path.insert(0, str(BACKEND_DIR))


async def run_all(dimensions: list[str] | None = None):
    """Run specified evaluation dimensions (or all if None)."""
    all_dims = ["scoring", "splitting", "expansion", "efficiency"]
    to_run = dimensions if dimensions else all_dims

    print("=" * 70)
    print("  AGI TREE — EVALUATION SUITE")
    print(f"  Dimensions: {', '.join(to_run)}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_results = []

    if "scoring" in to_run:
        try:
            from eval_scoring import evaluate_scoring
            results = await evaluate_scoring()
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n  ⚠ Scoring: {e}")
        except Exception as e:
            print(f"\n  ✗ Scoring failed: {e}")

    if "splitting" in to_run:
        try:
            from eval_splitting import evaluate_splitting
            results = await evaluate_splitting()
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n  ⚠ Splitting: {e}")
        except Exception as e:
            print(f"\n  ✗ Splitting failed: {e}")

    if "expansion" in to_run:
        try:
            from eval_expansion import evaluate_expansion
            results = await evaluate_expansion()
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n  ⚠ Expansion: {e}")
        except Exception as e:
            print(f"\n  ✗ Expansion failed: {e}")

    if "efficiency" in to_run:
        try:
            from eval_efficiency import evaluate_efficiency
            results = await evaluate_efficiency()
            all_results.extend(results)
        except FileNotFoundError as e:
            print(f"\n  ⚠ Efficiency: {e}")
        except Exception as e:
            print(f"\n  ✗ Efficiency failed: {e}")

    # --- Combined Report ---
    if all_results:
        print("\n" + "=" * 70)
        print("  COMBINED RESULTS")
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
        print(f"\n  Overall: {total_passed}/{total} tests passed "
              f"({total_passed/total*100:.0f}%)" if total > 0 else "")
        print("=" * 70)

        # Save combined report
        save_results(all_results, "combined_results.json")

    return all_results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")

    dims = sys.argv[1:] if len(sys.argv) > 1 else None
    asyncio.run(run_all(dims))
