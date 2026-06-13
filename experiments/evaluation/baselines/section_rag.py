"""
Section-RAG Baseline (Method C)
-------------------------------
Structure-aware but non-agentic baseline:
  1. Use DAT's section detection to split document into structural units
  2. Embed sections
  3. Retrieve top-k sections by similarity
  4. Generate answer from retrieved sections
  NO node self-scoring, NO node-specific prompts, NO lazy expansion, NO orchestration.
"""
import asyncio
import time
import numpy as np
from dataclasses import dataclass, field

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


async def run_section_rag(
    document_text: str,
    query: str,
    config: dict,
    doc_id: str = "",
) -> dict:
    """
    Run Section-RAG pipeline on a single query.
    """
    from llm_client import (
        classify_document, detect_top_level_nodes, get_answer, DEFAULT_MODEL
    )
    from baselines.flat_rag import embed_texts, cosine_similarity

    start_time = time.time()
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    errors = []

    top_k = config.get("section_rag", {}).get("top_k", 4)
    answer_model = config.get("models", {}).get("answer_model", DEFAULT_MODEL)
    embedding_model = config.get("models", {}).get("embedding_model", "text-embedding-3-small")

    # 1. Classify document
    try:
        classification = await classify_document(document_text)
        doc_type = classification.get("doc_type", "other")
        usage["llm_calls"] += 1
        usage["input_tokens"] += len(document_text[:5000]) // 4
        usage["output_tokens"] += 50
    except Exception as e:
        doc_type = "other"
        errors.append(f"Classification error: {e}")

    # 2. Detect sections using DAT's section detection
    try:
        sections, method = await detect_top_level_nodes(document_text, doc_type)
        usage["llm_calls"] += 1
        usage["input_tokens"] += len(document_text[:50000]) // 4
        usage["output_tokens"] += 500
    except Exception as e:
        sections = [{"name": "Full Document", "text": document_text}]
        errors.append(f"Section detection error: {e}")

    # 3. Embed sections + query
    try:
        section_texts = [s["text"][:3000] for s in sections]
        all_texts = section_texts + [query]
        embeddings, emb_usage = await embed_texts(all_texts, model=embedding_model)
        section_embeddings = embeddings[:-1]
        query_embedding = embeddings[-1]
        usage["llm_calls"] += emb_usage.get("embedding_calls", 1)
        usage["input_tokens"] += emb_usage.get("embedding_tokens", 0)
    except Exception as e:
        errors.append(f"Embedding error: {e}")
        return _make_result(query, "", [], usage, time.time() - start_time, errors, doc_id)

    # 4. Retrieve top-k sections
    similarities = []
    for i, emb in enumerate(section_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_sections = similarities[:top_k]

    selected_evidence = []
    context_parts = []
    for idx, sim in top_sections:
        section = sections[idx]
        selected_evidence.append({
            "unit_id": f"section_{idx:02d}",
            "unit_type": "section",
            "title": section["name"],
            "text": section["text"][:500],
            "score": round(sim, 4),
            "rationale": "",
        })
        context_parts.append(f"## {section['name']}\n\n{section['text']}")

    # 5. Generate answer (generic prompt, no node-specific prompts)
    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "You are a document question-answering assistant. "
        "Answer the question based ONLY on the provided document sections. "
        "If the sections do not contain enough information, say so clearly."
    )
    try:
        answer = await get_answer(system_prompt, query, context, model=answer_model)
        usage["llm_calls"] += 1
        usage["input_tokens"] += len(context) // 4 + len(query) // 4
        usage["output_tokens"] += len(answer) // 4
    except Exception as e:
        answer = f"Error generating answer: {e}"
        errors.append(str(e))

    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    latency = time.time() - start_time
    usage["latency_seconds"] = round(latency, 3)

    return {
        "method": "section_rag",
        "query": query,
        "doc_id": doc_id,
        "answer": answer,
        "selected_evidence": selected_evidence,
        "dat_trace": None,
        "usage": usage,
        "errors": errors,
    }


def _make_result(query, answer, evidence, usage, latency, errors, doc_id):
    usage["latency_seconds"] = round(latency, 3)
    usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return {
        "method": "section_rag",
        "query": query,
        "doc_id": doc_id,
        "answer": answer,
        "selected_evidence": evidence,
        "dat_trace": None,
        "usage": usage,
        "errors": errors,
    }
