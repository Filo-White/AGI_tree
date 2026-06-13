"""
Flat RAG Baseline (Method B)
----------------------------
Standard retrieval-augmented generation:
  1. Split document into fixed-size overlapping chunks
  2. Embed chunks using an embedding model
  3. Retrieve top-k chunks by cosine similarity to query embedding
  4. Generate answer from retrieved chunks using the same LLM as DAT
"""
import asyncio
import time
import re
import numpy as np
from dataclasses import dataclass, field

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


@dataclass
class RAGChunk:
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    embedding: list[float] = field(default_factory=list)
    parent_section: str = ""


@dataclass
class RAGResult:
    answer: str
    chunks: list[dict]
    usage: dict = field(default_factory=dict)


def chunk_document(text: str, chunk_size: int = 900, overlap: int = 120) -> list[RAGChunk]:
    """Split document into overlapping chunks by approximate token count."""
    # Approximate: 1 token ~ 4 chars for English, ~3 for Italian
    char_size = chunk_size * 4
    char_overlap = overlap * 4

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + char_size, len(text))
        # Try to break at sentence boundary
        if end < len(text):
            last_period = text.rfind('.', start + char_size // 2, end)
            if last_period > start:
                end = last_period + 1
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(RAGChunk(
                chunk_id=f"chunk_{idx:03d}",
                text=chunk_text,
                start_char=start,
                end_char=end,
            ))
            idx += 1
        start = end - char_overlap
        if start >= len(text) - 100:
            break

    return chunks


async def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> tuple[list[list[float]], dict]:
    """Embed a list of texts. Returns embeddings and usage stats."""
    from openai import AsyncOpenAI
    import os
    from dotenv import load_dotenv
    load_dotenv(BACKEND_DIR / ".env")

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    usage = {"embedding_calls": 0, "embedding_tokens": 0}
    all_embeddings = []

    # Batch in groups of 100
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(model=model, input=batch)
        for item in response.data:
            all_embeddings.append(item.embedding)
        usage["embedding_calls"] += 1
        usage["embedding_tokens"] += response.usage.total_tokens

    return all_embeddings, usage


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))


async def run_flat_rag(
    document_text: str,
    query: str,
    config: dict,
    doc_id: str = "",
) -> dict:
    """
    Run Flat RAG pipeline on a single query.
    Returns the standard logging schema dict.
    """
    from llm_client import get_answer, DEFAULT_MODEL

    start_time = time.time()
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    errors = []

    chunk_size = config.get("rag", {}).get("chunk_size_tokens", 900)
    chunk_overlap = config.get("rag", {}).get("chunk_overlap_tokens", 120)
    top_k = config.get("rag", {}).get("top_k", 4)
    answer_model = config.get("models", {}).get("answer_model", DEFAULT_MODEL)
    embedding_model = config.get("models", {}).get("embedding_model", "text-embedding-3-small")

    # 1. Chunk document
    chunks = chunk_document(document_text, chunk_size=chunk_size, overlap=chunk_overlap)

    # 2. Embed chunks + query
    try:
        all_texts = [c.text for c in chunks] + [query]
        embeddings, emb_usage = await embed_texts(all_texts, model=embedding_model)
        chunk_embeddings = embeddings[:-1]
        query_embedding = embeddings[-1]
        usage["llm_calls"] += emb_usage["embedding_calls"]
        usage["input_tokens"] += emb_usage["embedding_tokens"]
        usage["total_tokens"] += emb_usage["embedding_tokens"]
    except Exception as e:
        errors.append(f"Embedding error: {e}")
        return _make_result(query, "", [], usage, time.time() - start_time, errors, doc_id)

    # 3. Retrieve top-k
    similarities = []
    for i, emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_chunks = similarities[:top_k]

    selected_evidence = []
    context_parts = []
    for idx, sim in top_chunks:
        chunk = chunks[idx]
        selected_evidence.append({
            "unit_id": chunk.chunk_id,
            "unit_type": "chunk",
            "title": f"Chunk {idx}",
            "text": chunk.text[:500],
            "score": round(sim, 4),
            "rationale": "",
        })
        context_parts.append(chunk.text)

    # 4. Generate answer
    context = "\n\n---\n\n".join(context_parts)
    system_prompt = (
        "You are a document question-answering assistant. "
        "Answer the question based ONLY on the provided document excerpts. "
        "If the excerpts do not contain enough information, say so clearly."
    )
    try:
        answer = await get_answer(system_prompt, query, context, model=answer_model)
        usage["llm_calls"] += 1
        # Approximate token count
        usage["input_tokens"] += len(context) // 4 + len(query) // 4
        usage["output_tokens"] += len(answer) // 4
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    except Exception as e:
        answer = f"Error generating answer: {e}"
        errors.append(str(e))

    latency = time.time() - start_time

    return _make_result(query, answer, selected_evidence, usage, latency, errors, doc_id)


def _make_result(query, answer, evidence, usage, latency, errors, doc_id):
    usage["latency_seconds"] = round(latency, 3)
    return {
        "method": "flat_rag",
        "query": query,
        "doc_id": doc_id,
        "answer": answer,
        "selected_evidence": evidence,
        "dat_trace": None,
        "usage": usage,
        "errors": errors,
    }
