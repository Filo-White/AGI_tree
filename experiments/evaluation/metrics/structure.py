"""
Structural Processing Metrics — Document ingestion quality.
"""


def compute_structural_metrics(
    doc_id: str,
    doc_type: str,
    expected_type: str,
    predicted_type: str,
    expected_sections: list[str],
    generated_sections: list[str],
    document_text: str = "",
    nodes_text_total: int = 0,
) -> dict:
    """Compute structural metrics for a single document."""

    type_correct = predicted_type == expected_type

    expected_n = len(expected_sections)
    generated_n = len(generated_sections)
    node_count_error = abs(generated_n - expected_n)
    relative_error = node_count_error / expected_n if expected_n > 0 else 0.0

    # Section name coverage (fuzzy)
    def normalize(s):
        return s.lower().strip().replace("_", " ").replace("-", " ")

    matched = 0
    for exp in expected_sections:
        exp_norm = normalize(exp)
        if any(exp_norm in normalize(g) or normalize(g) in exp_norm for g in generated_sections):
            matched += 1
    name_coverage = matched / len(expected_sections) if expected_sections else 0.0

    # Text coverage
    text_coverage = min(nodes_text_total / len(document_text), 1.0) if document_text else 0.0

    # Granularity error
    if generated_n > expected_n * 1.5:
        granularity = "over_segmentation"
    elif generated_n < expected_n * 0.6:
        granularity = "under_segmentation"
    elif not type_correct:
        granularity = "wrong_level"
    else:
        granularity = "none"

    return {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "expected_type": expected_type,
        "predicted_type": predicted_type,
        "type_correct": type_correct,
        "expected_nodes": expected_n,
        "generated_nodes": generated_n,
        "node_count_error": node_count_error,
        "relative_error": round(relative_error, 3),
        "section_name_coverage": round(name_coverage, 3),
        "text_coverage": round(text_coverage, 3),
        "granularity_error": granularity,
        "notes": "",
    }
