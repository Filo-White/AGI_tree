"""
Long-Context / Full-Document Prompting Baseline (Method D)
----------------------------------------------------------
Monolithic baseline:
  1. Put the entire document in the prompt
  2. Ask the LLM to answer based only on the document
  3. If the document does not fit, truncate with a documented policy
"""
import time

import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Max chars to include (~128K tokens * 4 chars/token, conservative)
MAX_DOCUMENT_CHARS = 400000


async def run_long_context(
    document_text: str,
    query: str,
    config: dict,
    doc_id: str = "",
) -> dict:
    """
    Run long-context full-document prompting on a single query.
    """
    from llm_client import get_answer, DEFAULT_MODEL

    start_time = time.time()
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    errors = []

    answer_model = config.get("models", {}).get("answer_model", DEFAULT_MODEL)

    # Truncation policy
    truncated = False
    doc_for_prompt = document_text
    if len(document_text) > MAX_DOCUMENT_CHARS:
        doc_for_prompt = document_text[:MAX_DOCUMENT_CHARS]
        truncated = True
        errors.append(f"Document truncated from {len(document_text)} to {MAX_DOCUMENT_CHARS} chars")

    system_prompt = (
        "You are a document question-answering assistant. "
        "You have access to the FULL document below. "
        "Answer the question based ONLY on the document content. "
        "If the document does not contain enough information to answer, say so clearly. "
        "Do not make up information."
    )

    context = f"=== FULL DOCUMENT ===\n\n{doc_for_prompt}\n\n=== END OF DOCUMENT ==="

    try:
        answer = await get_answer(system_prompt, query, context, model=answer_model)
        usage["llm_calls"] = 1
        usage["input_tokens"] = len(context) // 4 + len(query) // 4 + len(system_prompt) // 4
        usage["output_tokens"] = len(answer) // 4
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    except Exception as e:
        answer = f"Error: {e}"
        errors.append(str(e))

    latency = time.time() - start_time
    usage["latency_seconds"] = round(latency, 3)

    return {
        "method": "long_context",
        "query": query,
        "doc_id": doc_id,
        "answer": answer,
        "selected_evidence": [{
            "unit_id": "full_document",
            "unit_type": "full_document",
            "title": "Full Document",
            "text": doc_for_prompt[:500],
            "score": 1.0,
            "rationale": "Full document provided" + (" (truncated)" if truncated else ""),
        }],
        "dat_trace": None,
        "usage": usage,
        "errors": errors,
    }
