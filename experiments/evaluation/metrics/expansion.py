"""
Lazy Expansion Metrics.
"""


def compute_expansion_metrics(
    expansion_triggered: bool,
    requires_expansion: bool,
    expanded_node: str = None,
    supporting_sections: list[str] = None,
    answer_quality_before: dict = None,
    answer_quality_after: dict = None,
    usage_before: dict = None,
    usage_after: dict = None,
) -> dict:
    """Compute expansion metrics for a single query."""

    # Classification
    if expansion_triggered and requires_expansion:
        classification = "true_positive"
    elif expansion_triggered and not requires_expansion:
        classification = "false_positive"
    elif not expansion_triggered and requires_expansion:
        classification = "false_negative"
    else:
        classification = "true_negative"

    # Quality delta
    quality_delta = None
    if answer_quality_before and answer_quality_after:
        before_score = answer_quality_before.get("correctness", 0) or 0
        after_score = answer_quality_after.get("correctness", 0) or 0
        quality_delta = after_score - before_score

    # Cost delta
    token_delta = None
    latency_delta = None
    if usage_before and usage_after:
        token_delta = (usage_after.get("total_tokens", 0) or 0) - (usage_before.get("total_tokens", 0) or 0)
        latency_delta = (usage_after.get("latency_seconds", 0) or 0) - (usage_before.get("latency_seconds", 0) or 0)

    return {
        "expansion_triggered": expansion_triggered,
        "requires_expansion": requires_expansion,
        "classification": classification,
        "expanded_node": expanded_node,
        "quality_delta": quality_delta,
        "token_overhead": token_delta,
        "latency_overhead": latency_delta,
    }


def compute_expansion_summary(all_expansion: list[dict]) -> dict:
    """Aggregate expansion metrics."""
    tp = sum(1 for e in all_expansion if e["classification"] == "true_positive")
    fp = sum(1 for e in all_expansion if e["classification"] == "false_positive")
    fn = sum(1 for e in all_expansion if e["classification"] == "false_negative")
    tn = sum(1 for e in all_expansion if e["classification"] == "true_negative")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    quality_deltas = [e["quality_delta"] for e in all_expansion if e["quality_delta"] is not None]
    token_overheads = [e["token_overhead"] for e in all_expansion if e["token_overhead"] is not None]
    latency_overheads = [e["latency_overhead"] for e in all_expansion if e["latency_overhead"] is not None]

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "avg_quality_delta": round(sum(quality_deltas) / len(quality_deltas), 3) if quality_deltas else None,
        "avg_token_overhead": round(sum(token_overheads) / len(token_overheads), 1) if token_overheads else None,
        "avg_latency_overhead": round(sum(latency_overheads) / len(latency_overheads), 3) if latency_overheads else None,
    }
