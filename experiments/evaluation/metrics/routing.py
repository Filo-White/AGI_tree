"""
Evidence Localization / Routing Quality Metrics.
"""


def compute_routing_metrics(
    selected_evidence: list[dict],
    supporting_sections: list[str],
) -> dict:
    """
    Compute routing metrics for a single query.

    Args:
        selected_evidence: list of evidence units selected by the method
        supporting_sections: gold supporting sections from annotation

    Returns:
        dict with hit@1, hit@3, mrr, recall, precision, etc.
    """
    if not supporting_sections:
        return {
            "hit_at_1": None, "hit_at_3": None, "mrr": None,
            "recall": None, "precision": None,
        }

    # Normalize titles for matching
    def normalize(s):
        return s.lower().strip().replace("_", " ").replace("-", " ")

    gold_set = {normalize(s) for s in supporting_sections}
    ranked_titles = [normalize(e.get("title", "")) for e in selected_evidence]

    # Hit@1
    hit_at_1 = 1 if ranked_titles and any(
        any(g in ranked_titles[0] or ranked_titles[0] in g for g in gold_set)
        for _ in [1]
    ) else 0

    # Hit@3
    top3 = ranked_titles[:3]
    hit_at_3 = 1 if any(
        any(g in t or t in g for g in gold_set)
        for t in top3
    ) else 0

    # MRR
    mrr = 0.0
    for i, t in enumerate(ranked_titles):
        if any(g in t or t in g for g in gold_set):
            mrr = 1.0 / (i + 1)
            break

    # Recall@k (how many gold sections were retrieved)
    retrieved_gold = set()
    for t in ranked_titles:
        for g in gold_set:
            if g in t or t in g:
                retrieved_gold.add(g)
    recall = len(retrieved_gold) / len(gold_set) if gold_set else 0.0

    # Precision@k
    relevant_count = sum(
        1 for t in ranked_titles
        if any(g in t or t in g for g in gold_set)
    )
    precision = relevant_count / len(ranked_titles) if ranked_titles else 0.0

    return {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": round(mrr, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
    }


def compute_dat_score_separation(
    node_scores: list[dict],
    supporting_sections: list[str],
) -> dict:
    """Compute DAT-specific score separation metrics."""
    if not node_scores or not supporting_sections:
        return {
            "avg_relevant_score": None,
            "avg_irrelevant_score": None,
            "score_separation": None,
            "correct_node_selected": None,
        }

    def normalize(s):
        return s.lower().strip()

    gold_set = {normalize(s) for s in supporting_sections}

    relevant_scores = []
    irrelevant_scores = []
    for ns in node_scores:
        name = normalize(ns.get("name", ns.get("node_id", "")))
        score = ns.get("score", 0.0)
        if any(g in name or name in g for g in gold_set):
            relevant_scores.append(score)
        else:
            irrelevant_scores.append(score)

    avg_rel = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0
    avg_irr = sum(irrelevant_scores) / len(irrelevant_scores) if irrelevant_scores else 0.0

    return {
        "avg_relevant_score": round(avg_rel, 4),
        "avg_irrelevant_score": round(avg_irr, 4),
        "score_separation": round(avg_rel - avg_irr, 4),
        "correct_node_selected": len(relevant_scores) > 0 and max(relevant_scores, default=0) >= max(irrelevant_scores, default=0),
    }


def compute_routing_summary(all_routing: list[dict]) -> dict:
    """Aggregate routing metrics across queries."""
    metrics = ["hit_at_1", "hit_at_3", "mrr", "recall", "precision"]
    summary = {}
    for m in metrics:
        values = [r[m] for r in all_routing if r.get(m) is not None]
        if values:
            summary[f"{m}_mean"] = round(sum(values) / len(values), 4)
        else:
            summary[f"{m}_mean"] = None
    summary["n_queries"] = len(all_routing)
    return summary
