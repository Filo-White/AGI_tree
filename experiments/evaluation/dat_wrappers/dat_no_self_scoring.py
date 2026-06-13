"""
DAT-no-self-scoring Ablation — Replace LLM competence scoring with embedding similarity.
"""
import time
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


async def run_dat_no_self_scoring(
    document_text: str,
    query: str,
    config: dict,
    doc_id: str = "",
) -> dict:
    """DAT with embedding-based retrieval instead of LLM self-scoring."""
    from tree_engine import TreeEngine
    from baselines.flat_rag import embed_texts, cosine_similarity

    start_time = time.time()
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    errors = []
    embedding_model = config.get("models", {}).get("embedding_model", "text-embedding-3-small")

    engine = TreeEngine()

    try:
        await engine.build_tree_from_document(document_text, filename=doc_id)
        usage["llm_calls"] += 2 + len(engine.root.children)
        usage["input_tokens"] += len(document_text[:50000]) // 4
        usage["output_tokens"] += 500 * len(engine.root.children)
    except Exception as e:
        errors.append(f"Tree build error: {e}")
        usage["latency_seconds"] = round(time.time() - start_time, 3)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return {"method": "dat_no_self_scoring", "query": query, "doc_id": doc_id,
                "answer": "", "selected_evidence": [], "dat_trace": None,
                "usage": usage, "errors": errors}

    # Replace scoring with embedding similarity
    nodes = []
    for child in engine.root.children:
        if child.children:
            for leaf in child.children:
                nodes.append(leaf)
        else:
            nodes.append(child)

    try:
        node_texts = [f"{n.name}: {n.context[:2000]}" for n in nodes]
        all_texts = node_texts + [query]
        embeddings, emb_usage = await embed_texts(all_texts, model=embedding_model)
        query_emb = embeddings[-1]
        node_embs = embeddings[:-1]
        usage["llm_calls"] += emb_usage.get("embedding_calls", 1)
        usage["input_tokens"] += emb_usage.get("embedding_tokens", 0)

        # Compute similarity scores
        scores = {}
        reasons = {}
        for i, node in enumerate(nodes):
            sim = cosine_similarity(query_emb, node_embs[i])
            scores[node.id] = sim
            reasons[node.id] = f"embedding similarity: {sim:.3f}"
    except Exception as e:
        errors.append(f"Embedding scoring error: {e}")
        usage["latency_seconds"] = round(time.time() - start_time, 3)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return {"method": "dat_no_self_scoring", "query": query, "doc_id": doc_id,
                "answer": "", "selected_evidence": [], "dat_trace": None,
                "usage": usage, "errors": errors}

    # Select top responders (same logic as DAT)
    selected = engine.select_responders(scores)

    # Route and respond (uses DAT's answer generation + synthesis)
    try:
        final_response, sub_queries, leaf_responses = await engine.route_and_respond(
            query, selected, scores
        )
        n_selected = len(selected)
        usage["llm_calls"] += n_selected + 2  # answering + decompose + synthesis
        usage["input_tokens"] += n_selected * 1200
        usage["output_tokens"] += n_selected * 500 + 800
    except Exception as e:
        final_response = f"Error: {e}"
        sub_queries = []
        leaf_responses = []
        errors.append(str(e))

    selected_evidence = []
    for nid in selected:
        node = engine.get_node(nid)
        if node:
            selected_evidence.append({
                "unit_id": nid, "unit_type": node.role,
                "title": node.name, "text": node.context[:500],
                "score": round(scores.get(nid, 0.0), 4),
                "rationale": reasons.get(nid, ""),
            })

    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    usage["latency_seconds"] = round(time.time() - start_time, 3)

    return {
        "method": "dat_no_self_scoring", "query": query, "doc_id": doc_id,
        "answer": final_response,
        "selected_evidence": selected_evidence,
        "dat_trace": {
            "predicted_doc_type": engine.processing_log.get("doc_type", ""),
            "scorable_nodes": list(scores.keys()),
            "node_scores": [{"node_id": k, "score": round(v, 4)} for k, v in scores.items()],
            "selected_nodes": selected,
            "expansion_triggered": False, "expanded_node": None,
            "query_decomposition": sub_queries,
            "local_answers": [],
        },
        "usage": usage, "errors": errors,
    }
