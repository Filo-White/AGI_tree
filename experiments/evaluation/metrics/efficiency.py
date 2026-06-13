"""
Efficiency Metrics — Cost, latency, token usage.
"""


def compute_efficiency_metrics(usage: dict, config: dict = None) -> dict:
    """Compute efficiency metrics for a single query from usage dict."""
    costs = (config or {}).get("costs", {})
    input_cost_per_1k = costs.get("answer_model_input_per_1k", 0.00015)
    output_cost_per_1k = costs.get("answer_model_output_per_1k", 0.0006)

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    estimated_cost = (input_tokens / 1000) * input_cost_per_1k + (output_tokens / 1000) * output_cost_per_1k

    return {
        "llm_calls": usage.get("llm_calls", 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "latency_seconds": usage.get("latency_seconds", 0.0),
        "estimated_cost_usd": round(estimated_cost, 6),
    }


def compute_efficiency_summary(all_efficiency: list[dict], method: str) -> dict:
    """Aggregate efficiency metrics for a method."""
    n = len(all_efficiency)
    if n == 0:
        return {"method": method, "n_queries": 0}

    def avg(key):
        values = [e.get(key, 0) for e in all_efficiency]
        return round(sum(values) / len(values), 2) if values else 0

    def total(key):
        return sum(e.get(key, 0) for e in all_efficiency)

    return {
        "method": method,
        "n_queries": n,
        "avg_llm_calls": avg("llm_calls"),
        "avg_input_tokens": avg("input_tokens"),
        "avg_output_tokens": avg("output_tokens"),
        "avg_total_tokens": avg("total_tokens"),
        "avg_latency_seconds": avg("latency_seconds"),
        "avg_estimated_cost_usd": avg("estimated_cost_usd"),
        "total_tokens": total("total_tokens"),
        "total_cost_usd": round(total("estimated_cost_usd"), 4),
    }
