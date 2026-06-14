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


def _measure_doc_length(doc_path: Path) -> int:
    """Return character count for a document."""
    if doc_path.suffix == ".txt":
        return len(doc_path.read_text(encoding="utf-8"))
    elif doc_path.suffix == ".pdf":
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            from pypdf import PdfReader
        reader = PdfReader(str(doc_path))
        return sum(len(page.extract_text() or "") for page in reader.pages)
    return 0


def make_table_dataset():
    """Table 1: Dataset overview."""
    queries_file = EVAL_DIR / "data" / "queries.jsonl"
    docs_file = EVAL_DIR / "data" / "documents.json"
    base_dir = EVAL_DIR.parent.parent  # repo root

    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    with open(docs_file, "r", encoding="utf-8") as f:
        docs = json.load(f)

    doc_type_map = {d["doc_id"]: d["doc_type"] for d in docs}
    doc_path_map = {d["doc_id"]: d["path"] for d in docs}

    # Check which docs are controlled (from evaluation/test_data)
    controlled_ids = {d["doc_id"] for d in docs if "test_data" in d.get("path", "")}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Dataset overview. Documents marked with $\dagger$ are author-created",
        r" with known ground-truth structure; all others are publicly available real-world documents.}",
        r"\label{tab:dataset}",
        r"\begin{tabular}{llcrrr}",
        r"\toprule",
        r"\textbf{Doc ID} & \textbf{Title} & \textbf{Type} & \textbf{Length (k\,ch)} & \textbf{Queries} & \textbf{Language} \\",
        r"\midrule",
    ]

    total_queries = 0
    all_lengths = []
    for d in docs:
        doc_id = d["doc_id"]
        title = d["title"]
        if len(title) > 42:
            title = title[:39] + "..."
        mark = "$^\\dagger$" if doc_id in controlled_ids else ""
        dtype_labels = {"textbook": "textbook", "scientific_paper": "paper",
                        "administrative": "admin.", "report": "report",
                        "regulation": "regulation", "manual": "manual"}
        dtype = dtype_labels.get(d["doc_type"], d["doc_type"])
        n_q = sum(1 for q in queries if q["doc_id"] == doc_id)
        total_queries += n_q
        p = base_dir / d["path"]
        length = _measure_doc_length(p) if p.exists() else 0
        all_lengths.append(length)
        len_k = f"{length / 1000:.1f}"
        lang = d.get("language", "en").upper()
        lines.append(f"{doc_id} & {title}{mark} & {dtype} & {len_k} & {n_q} & {lang} \\\\")

    total_len = f"{sum(all_lengths) / 1000:.1f}"
    lines.append(r"\midrule")
    lines.append(f"\\textbf{{Total}} & & & {total_len} & {total_queries} & \\\\")
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

    # Count unanswerable queries
    queries_file = EVAL_DIR / "data" / "queries.jsonl"
    n_unanswerable = 0
    if queries_file.exists():
        with open(queries_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    if not q.get("answerable", True):
                        n_unanswerable += 1

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{End-to-end answer quality. Scores are LLM-judge ratings on a 1--5 Likert scale"
        r" (higher is better). Abstention accuracy is computed on the $n=" + str(n_unanswerable) + r"$"
        r" unanswerable queries. Judge: \texttt{gpt-5.4-nano-2026-03-17}, $T{=}0$, method names anonymized.}",
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
        r"\caption{Evidence localization and routing quality. Long-Context is marked N/A"
        r" because it feeds the entire document as a single prompt and does not"
        r" perform discrete evidence selection.}",
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
        r"\caption{Computational efficiency per query. Costs estimated using"
        r" \texttt{gpt-5.4-nano-2026-03-17} pricing"
        r" (\$0.15/M input, \$0.60/M output tokens).}",
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
        "textbook": "Controlled",
        "scientific_paper": "Scientific Papers",
        "administrative": "Administrative/Legal",
        "regulation": "Regulation",
        "report": "Reports",
        "manual": "Manuals/Guides",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Structural processing quality by document category."
        r" Administrative/legal documents exhibit systematic over-segmentation"
        r" due to misalignment between their nested Title$\to$Article$\to$Paragraph"
        r" hierarchy and DAT's two-level node/leaf architecture.}",
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


def make_table_bootstrap():
    """Table 7: Bootstrap confidence intervals."""
    rows = read_csv(RESULTS_DIR / "bootstrap_confidence_intervals.csv")
    if not rows:
        return "% No bootstrap data available"

    method_labels = {
        "flat_rag": "Flat RAG",
        "section_rag": "Section-RAG",
        "long_context": "Long-Context",
        "dat_full": "DAT Full",
    }
    method_order = ["flat_rag", "section_rag", "long_context", "dat_full"]
    method_rows = {r.get("method"): r for r in rows}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Bootstrap 95\% confidence intervals (10{,}000 resamples, $n=122$).}",
        r"\label{tab:bootstrap}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"\textbf{Method} & \textbf{Correctness (95\% CI)} & \textbf{Faithfulness (95\% CI)} \\",
        r"\midrule",
    ]

    for m in method_order:
        label = method_labels.get(m, m)
        r = method_rows.get(m, {})
        c_mean = fmt(r.get("correctness_mean"))
        c_lo = fmt(r.get("correctness_ci_lo"))
        c_hi = fmt(r.get("correctness_ci_hi"))
        f_mean = fmt(r.get("faithfulness_mean"))
        f_lo = fmt(r.get("faithfulness_ci_lo"))
        f_hi = fmt(r.get("faithfulness_ci_hi"))
        lines.append(f"{label} & {c_mean} [{c_lo}, {c_hi}] & {f_mean} [{f_lo}, {f_hi}] \\\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def make_table_inter_judge():
    """Table 8: Inter-judge agreement."""
    rows = read_csv(RESULTS_DIR / "inter_judge_agreement.csv")
    if not rows:
        return "% No inter-judge data available"

    # Load correlation
    corr_file = RESULTS_DIR / "inter_judge_correlation.json"
    spearman = "---"
    if corr_file.exists():
        with open(corr_file, "r", encoding="utf-8") as f:
            corr = json.load(f)
            spearman = fmt(corr.get("spearman_rho"), 3)

    method_labels = {
        "flat_rag": "Flat RAG",
        "section_rag": "Section-RAG",
        "long_context": "Long-Context",
        "dat_full": "DAT Full",
    }
    method_order = ["flat_rag", "section_rag", "long_context", "dat_full"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Inter-judge agreement. Primary: \texttt{gpt-5.4-nano}, secondary:",
        r" \texttt{deepseek-v4-pro}. Spearman $\rho = " + spearman + r"$ on method ranking.}",
        r"\label{tab:inter_judge}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Method} & \multicolumn{2}{c}{\textbf{Primary Judge}} & \multicolumn{2}{c}{\textbf{Secondary Judge}} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r" & Correct. & Faithful. & Correct. & Faithful. \\",
        r"\midrule",
    ]

    for m in method_order:
        label = method_labels.get(m, m)
        primary = [r for r in rows if r["method"] == m and r["judge"] == "primary"]
        secondary = [r for r in rows if r["method"] == m and r["judge"] == "secondary"]
        p = primary[0] if primary else {}
        s = secondary[0] if secondary else {}
        lines.append(
            f"{label} & {fmt(p.get('correctness_mean'))} & {fmt(p.get('faithfulness_mean'))}"
            f" & {fmt(s.get('correctness_mean'))} & {fmt(s.get('faithfulness_mean'))} \\\\")

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
        "table_bootstrap.tex": make_table_bootstrap,
        "table_inter_judge.tex": make_table_inter_judge,
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
