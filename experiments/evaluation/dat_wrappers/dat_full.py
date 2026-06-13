"""
DAT Full (Method A) — Full Document Agent Tree pipeline with logging.
"""
import time
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


async def run_dat_full(
    document_text: str,
    query: str,
    config: dict,
    doc_id: str = "",
) -> dict:
    """Run full DAT pipeline with detailed trace logging."""
    from tree_engine import TreeEngine
    from llm_client import classify_document

    start_time = time.time()
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    errors = []

    engine = TreeEngine()

    # Build tree
    try:
        await engine.build_tree_from_document(document_text, filename=doc_id)
        # Approximate usage for tree building
        usage["llm_calls"] += 2 + len(engine.root.children)  # classify + detect + prompts
        usage["input_tokens"] += len(document_text[:50000]) // 4
        usage["output_tokens"] += 500 * len(engine.root.children)
    except Exception as e:
        errors.append(f"Tree build error: {e}")
        return _make_empty_result(query, doc_id, usage, time.time() - start_time, errors)

    # Process query (includes scoring, auto-expansion, routing, answering, synthesis)
    try:
        result = await engine.process_query(query)
    except Exception as e:
        errors.append(f"Query processing error: {e}")
        return _make_empty_result(query, doc_id, usage, time.time() - start_time, errors)

    # Build evidence list
    selected_evidence = []
    for nid in result.get("selected_leaves", []):
        node = engine.get_node(nid)
        if node:
            selected_evidence.append({
                "unit_id": nid,
                "unit_type": "leaf" if node.role == "leaf" else "node",
                "title": node.name,
                "text": node.context[:500] if node.context else "",
                "score": result["scores"].get(nid, 0.0),
                "rationale": result["reasons"].get(nid, ""),
            })

    # Build DAT trace
    scorable_nodes = []
    node_scores = []
    for nid, score in result.get("scores", {}).items():
        node = engine.get_node(nid)
        if node:
            scorable_nodes.append(nid)
            node_scores.append({
                "node_id": nid,
                "name": node.name,
                "score": score,
                "reason": result["reasons"].get(nid, ""),
            })

    dat_trace = {
        "predicted_doc_type": engine.processing_log.get("doc_type", ""),
        "scorable_nodes": scorable_nodes,
        "node_scores": node_scores,
        "selected_nodes": result.get("selected_leaves", []),
        "expansion_triggered": result.get("auto_expanded") is not None,
        "expanded_node": result.get("auto_expanded"),
        "query_decomposition": result.get("sub_queries", []),
        "local_answers": [
            {"node_id": lr["node_id"], "node_name": lr["node_name"], "answer": lr["response"][:200]}
            for lr in result.get("leaf_responses", [])
        ],
    }

    # Approximate usage
    n_nodes = len(result.get("scores", {}))
    n_selected = len(result.get("selected_leaves", []))
    usage["llm_calls"] += n_nodes + n_selected + 2  # scoring + answering + decompose + synthesis
    usage["input_tokens"] += n_nodes * 800 + n_selected * 1200
    usage["output_tokens"] += n_nodes * 50 + n_selected * 500 + 800
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

    latency = time.time() - start_time
    usage["latency_seconds"] = round(latency, 3)

    return {
        "method": "dat_full",
        "query": query,
        "doc_id": doc_id,
        "answer": result.get("response", ""),
        "selected_evidence": selected_evidence,
        "dat_trace": dat_trace,
        "usage": usage,
        "errors": errors,
    }


def _make_empty_result(query, doc_id, usage, latency, errors):
    usage["latency_seconds"] = round(latency, 3)
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return {
        "method": "dat_full",
        "query": query,
        "doc_id": doc_id,
        "answer": "",
        "selected_evidence": [],
        "dat_trace": None,
        "usage": usage,
        "errors": errors,
    }
