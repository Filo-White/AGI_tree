"""
Generate LaTeX tables and CSV summaries from computed metrics.

Usage:
    python experiments/evaluation/make_tables.py --run-dir experiments/evaluation/outputs/runs/<RUN_ID>
    python experiments/evaluation/make_tables.py --from-csv   (uses results/ CSVs directly)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
LATEX_DIR = RESULTS_DIR / "latex_tables"


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(val, decimals=2):
    """Format a value for LaTeX."""
    if val is None or val == "" or val == "None":
        return "---"
    try:
        v = float(val)
        return f"{v:.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def bold_best(values: list[str], higher_better=True) -> list[str]:
    """Bold the best value in a list of formatted strings."""
    nums = []
    for v in values:
        try:
            nums.append(float(v.replace("---", "nan")))
        except ValueError:
            nums.append(float("nan"))

    import math
    valid = [(i, n) for i, n in enumerate(nums) if not math.isnan(n)]
    if not valid:
        return values

    if higher_better:
        best_idx = max(valid, key=lambda x: x[1])[0]
    else:
        best_idx = min(valid, key=lambda x: x[1])[0]

    result = list(values)
    result[best_idx] = f"\\textbf{{{result[best_idx]}}}"
    return result


def make_table_dataset():
    """Table 1: Dataset overview."""
    queries_file = EVAL_DIR / "data" / "queries.jsonl"
    docs_file = EVAL_DIR / "data" / "documents.json"

    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    categories = {
        "textbook": {"label": "Textbook", "docs": [], "queries": []},
        "scientific_paper": {"label": "Scientific Papers", "docs": [], "queries": []},
        "administrative": {"label": "Administrative/Legal", "docs": [], "queries": []},
        "report": {"label": "Reports", "docs": [], "queries": []},
    }

    doc_type_map = {d["doc_id"]: d["doc_type"] for d in docs}
    for d in docs:
        cat = d["doc_type"]
        if cat in categories:
            categories[cat]["docs"].append(d)

    for q in queries:
        cat = doc_type_map.get(q["doc_id"], "")
        if cat in categories:
            categories[cat]["queries"].append(q)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset overview.}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Docs} & \textbf{Queries} & \textbf{Avg. Length (chars)} & \textbf{Query Types} \\",
        r"\midrule",
    ]

    total_docs = 0
    total_queries = 0
    for cat_key, cat in categories.items():
        n_docs = len(cat["docs"])
        n_queries = len(cat["queries"])
        total_docs += n_docs
        total_queries += n_queries
        # Query type distribution
        qtypes = {}
        for q in cat["queries"]:
            qt = q["query_type"]
            qtypes[qt] = qtypes.get(qt, 0) + 1
        qt_str = ", ".join(f"{k[:3].upper()}: {v}" for k, v in sorted(qtypes.items()))
        avg_len = "---"  # Would need actual doc lengths
        lines.append(f"{cat['label']} & {n_docs} & {n_queries} & {avg_len} & {qt_str} \\\\")

    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & {total_docs} & {total_queries} & --- & --- \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    return "\n".join(lines)


def make_table_answer_quality():
    """Table 2: End-to-end answer quality."""
    rows = read_csv(RESULTS_DIR / "answer_quality_summary.csv")
    if not rows:
        return "% No answer quality data available"

    method_order = ["flat_rag", "section_rag", "long_context", "dat_full"]
    method_labels = {
        "flat_rag": "Flat RAG",
        "section_rag": "Section-RAG",
        "long_context": "Long-Context",
        "dat_full": "DAT Full",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{End-to-end answer quality (1--5 scale, higher is better).}",
        r"\label{tab:answer_quality}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Correct.} & \textbf{Faithful.} & \textbf{Complete.} & \textbf{Relev.} & \textbf{Abst. Acc.} \\",
        r"\midrule",
    ]

    method_rows = {r.get("method"): r for r in rows}
    cols = ["correctness_mean", "faithfulness_mean", "completeness_mean", "relevance_mean", "abstention_accuracy"]

    # Collect values for bolding
    col_values = {c: [] for c in cols}
    for m in method_order:
        r = method_rows.get(m, {})
        for c in cols:
            col_values[c].append(fmt(r.get(c)))

    for ci, c in enumerate(cols):
        col_values[c] = bold_best(col_values[c], higher_better=True)

    for mi, m in enumerate(method_order):
        label = method_labels.get(m, m)
        vals = " & ".join(col_values[c][mi] for c in cols)
        lines.append(f"{label} & {vals} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_table_routing():
    """Table 3: Evidence localization / routing."""
    rows = read_csv(RESULTS_DIR / "routing_metrics.csv")
    if not rows:
        return "% No routing data available"

    method_order = ["flat_rag", "section_rag", "long_context", "dat_full"]
    method_labels = {
        "flat_rag": "Flat RAG",
        "section_rag": "Section-RAG",
        "long_context": "Long-Context",
        "dat_full": "DAT Full",
    }

    # Aggregate by method
    from metrics.routing import compute_routing_summary
    method_summaries = {}
    for m in method_order:
        m_rows = [r for r in rows if r.get("method") == m]
        # Convert string values to float
        for r in m_rows:
            for k in ["hit_at_1", "hit_at_3", "mrr", "recall", "precision"]:
                try:
                    r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
                except (ValueError, TypeError):
                    r[k] = None
        method_summaries[m] = compute_routing_summary(m_rows)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Evidence localization and routing quality.}",
        r"\label{tab:routing}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Hit@1} & \textbf{Hit@3} & \textbf{MRR} & \textbf{Recall} \\",
        r"\midrule",
    ]

    for m in method_order:
        label = method_labels.get(m, m)
        s = method_summaries.get(m, {})
        if m == "long_context":
            lines.append(f"{label} & N/A & N/A & N/A & N/A \\\\")
        else:
            vals = " & ".join([
                fmt(s.get("hit_at_1_mean"), 3),
                fmt(s.get("hit_at_3_mean"), 3),
                fmt(s.get("mrr_mean"), 3),
                fmt(s.get("recall_mean"), 3),
            ])
            lines.append(f"{label} & {vals} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_table_efficiency():
    """Table 4: Efficiency comparison."""
    rows = read_csv(RESULTS_DIR / "efficiency_summary.csv")
    if not rows:
        return "% No efficiency data available"

    method_order = ["flat_rag", "section_rag", "long_context", "dat_full"]
    method_labels = {
        "flat_rag": "Flat RAG",
        "section_rag": "Section-RAG",
        "long_context": "Long-Context",
        "dat_full": "DAT Full",
    }

    method_rows = {r.get("method"): r for r in rows}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Computational efficiency per query.}",
        r"\label{tab:efficiency}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{LLM Calls} & \textbf{Tokens} & \textbf{Latency (s)} & \textbf{Cost (\$)} \\",
        r"\midrule",
    ]

    for m in method_order:
        label = method_labels.get(m, m)
        r = method_rows.get(m, {})
        vals = " & ".join([
            fmt(r.get("avg_llm_calls"), 1),
            fmt(r.get("avg_total_tokens"), 0),
            fmt(r.get("avg_latency_seconds"), 2),
            fmt(r.get("avg_estimated_cost_usd"), 4),
        ])
        lines.append(f"{label} & {vals} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_table_ablation():
    """Table 5: Ablation study."""
    rows = read_csv(RESULTS_DIR / "ablation_summary.csv")
    if not rows:
        return "% No ablation data available"

    method_order = ["dat_full", "dat_no_expansion", "dat_parent_only", "dat_no_self_scoring"]
    method_labels = {
        "dat_full": "DAT Full",
        "dat_no_expansion": "DAT w/o Expansion",
        "dat_parent_only": "DAT Parent-Only",
        "dat_no_self_scoring": "DAT w/o Self-Scoring",
    }

    method_rows = {r.get("method"): r for r in rows}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: impact of DAT components.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Correct.} & \textbf{Faithful.} & \textbf{Hit@3} & \textbf{Tokens/q} & \textbf{Latency (s)} \\",
        r"\midrule",
    ]

    for m in method_order:
        label = method_labels.get(m, m)
        r = method_rows.get(m, {})
        vals = " & ".join([
            fmt(r.get("correctness"), 2),
            fmt(r.get("faithfulness"), 2),
            fmt(r.get("hit_at_3"), 3),
            fmt(r.get("avg_total_tokens"), 0),
            fmt(r.get("avg_latency"), 2),
        ])
        lines.append(f"{label} & {vals} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_table_structure():
    """Table 6: Structural processing quality."""
    rows = read_csv(RESULTS_DIR / "structural_metrics.csv")
    if not rows:
        return "% No structural data available"

    categories = {
        "textbook": "Textbook",
        "scientific_paper": "Scientific Papers",
        "administrative": "Administrative/Legal",
        "report": "Reports",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Structural processing quality by document category.}",
        r"\label{tab:structure}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"\textbf{Category} & \textbf{Type Acc.} & \textbf{Node Err.} & \textbf{Name Cov.} & \textbf{Text Cov.} & \textbf{Gran. Err.} \\",
        r"\midrule",
    ]

    for cat_key, cat_label in categories.items():
        cat_rows = [r for r in rows if r.get("doc_type") == cat_key]
        if not cat_rows:
            lines.append(f"{cat_label} & --- & --- & --- & --- & --- \\\\")
            continue
        type_acc = sum(1 for r in cat_rows if r.get("type_correct") in (True, "True")) / len(cat_rows)
        avg_err = sum(float(r.get("node_count_error", 0)) for r in cat_rows) / len(cat_rows)
        avg_name = sum(float(r.get("section_name_coverage", 0)) for r in cat_rows) / len(cat_rows)
        avg_text = sum(float(r.get("text_coverage", 0)) for r in cat_rows) / len(cat_rows)
        gran_errs = sum(1 for r in cat_rows if r.get("granularity_error", "none") != "none")
        gran_rate = gran_errs / len(cat_rows)

        lines.append(f"{cat_label} & {type_acc:.2f} & {avg_err:.1f} & {avg_name:.2f} & {avg_text:.2f} & {gran_rate:.2f} \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_all_tables():
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    tables = {
        "table_dataset.tex": make_table_dataset,
        "table_answer_quality.tex": make_table_answer_quality,
        "table_routing.tex": make_table_routing,
        "table_efficiency.tex": make_table_efficiency,
        "table_ablation.tex": make_table_ablation,
        "table_structure.tex": make_table_structure,
    }

    for filename, generator in tables.items():
        try:
            content = generator()
            path = LATEX_DIR / filename
            with open(path, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            print(f"Generated {filename}")
        except Exception as e:
            print(f"Error generating {filename}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--from-csv", action="store_true", help="Use existing CSVs in results/")
    args = parser.parse_args()
    generate_all_tables()


if __name__ == "__main__":
    main()
