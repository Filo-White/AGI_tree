"""
DAT-parent-only Ablation — Only top-level nodes, no leaf expansion ever.
"""
import time
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


async def run_dat_parent_only(
    document_text: str,
    query: str,
    config: dict,
    doc_id: str = "",
) -> dict:
    """DAT with only parent nodes — expansion disabled, existing leaves ignored."""
    from tree_engine import TreeEngine

    start_time = time.time()
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    errors = []

    engine = TreeEngine()

    # Disable all expansion
    async def no_expand(*args, **kwargs):
        return False, None
    engine.maybe_auto_expand = no_expand

    async def no_expand_node(*args, **kwargs):
        return False
    engine.expand_node = no_expand_node

    try:
        await engine.build_tree_from_document(document_text, filename=doc_id)
        usage["llm_calls"] += 2 + len(engine.root.children)
        usage["input_tokens"] += len(document_text[:50000]) // 4
        usage["output_tokens"] += 500 * len(engine.root.children)
    except Exception as e:
        errors.append(f"Tree build error: {e}")

    try:
        result = await engine.process_query(query)
    except Exception as e:
        errors.append(f"Query error: {e}")
        usage["latency_seconds"] = round(time.time() - start_time, 3)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return {"method": "dat_parent_only", "query": query, "doc_id": doc_id,
                "answer": "", "selected_evidence": [], "dat_trace": None,
                "usage": usage, "errors": errors}

    selected_evidence = []
    for nid in result.get("selected_leaves", []):
        node = engine.get_node(nid)
        if node:
            selected_evidence.append({
                "unit_id": nid, "unit_type": "node", "title": node.name,
                "text": node.context[:500], "score": result["scores"].get(nid, 0.0),
                "rationale": result["reasons"].get(nid, ""),
            })

    n_nodes = len(result.get("scores", {}))
    n_selected = len(result.get("selected_leaves", []))
    usage["llm_calls"] += n_nodes + n_selected + 2
    usage["input_tokens"] += n_nodes * 800 + n_selected * 1200
    usage["output_tokens"] += n_nodes * 50 + n_selected * 500 + 800
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["latency_seconds"] = round(time.time() - start_time, 3)

    return {
        "method": "dat_parent_only", "query": query, "doc_id": doc_id,
        "answer": result.get("response", ""), "selected_evidence": selected_evidence,
        "dat_trace": {
            "predicted_doc_type": engine.processing_log.get("doc_type", ""),
            "scorable_nodes": list(result.get("scores", {}).keys()),
            "node_scores": [{"node_id": k, "score": v} for k, v in result.get("scores", {}).items()],
            "selected_nodes": result.get("selected_leaves", []),
            "expansion_triggered": False, "expanded_node": None,
            "query_decomposition": result.get("sub_queries", []),
            "local_answers": [],
        },
        "usage": usage, "errors": errors,
    }
