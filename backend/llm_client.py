import json
import os
import re

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

_client: AsyncOpenAI | None = None
DEFAULT_MODEL = "gpt-5.4-nano-2026-03-17"


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY non configurata. "
                "Crea un file backend/.env con OPENAI_API_KEY=sk-..."
            )
        _client = AsyncOpenAI(api_key=api_key)
    return _client


async def get_competence_score(system_prompt: str, query: str, model: str = DEFAULT_MODEL) -> float:
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Valuta la tua competenza sulla seguente domanda. "
                    "Rispondi SOLO con un numero decimale tra 0.0 e 1.0, dove "
                    "0.0 = nessuna competenza e 1.0 = massima competenza.\n\n"
                    f"Domanda: {query}\n\nScore:"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=10,
    )
    try:
        text = response.choices[0].message.content.strip()
        match = re.search(r"[0-1]\.?\d*", text)
        score = float(match.group()) if match else 0.0
        return max(0.0, min(1.0, score))
    except (ValueError, IndexError, AttributeError):
        return 0.0


async def get_answer(
    system_prompt: str,
    query: str,
    document_context: str | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    if document_context:
        user_content = (
            f"Contesto dal documento caricato:\n---\n{document_context}\n---\n\n"
            f"Domanda: {query}"
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})

    response = await _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
    )
    return response.choices[0].message.content


async def decompose_query(query: str, model: str = DEFAULT_MODEL) -> list[str]:
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei un analizzatore di query. Determina se una domanda contiene "
                    "più sotto-domande su argomenti distinti.\n"
                    "Se la domanda è singola o riguarda un unico argomento, restituisci solo quella.\n"
                    "Se contiene più argomenti diversi, scomponila in sotto-domande separate.\n\n"
                    'Rispondi SOLO in formato JSON: {"sub_queries": ["query1", "query2", ...]}'
                ),
            },
            {"role": "user", "content": query},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.removeprefix("```json").removesuffix("```").strip()
        result = json.loads(text)
        sub_queries = result.get("sub_queries", [query])
        return sub_queries if sub_queries else [query]
    except (json.JSONDecodeError, KeyError):
        return [query]


async def synthesize_response(
    query: str, leaf_responses: list[dict], model: str = DEFAULT_MODEL
) -> str:
    parts = []
    for resp in sorted(leaf_responses, key=lambda r: r["score"], reverse=True):
        parts.append(
            f"[Esperto: {resp['name']} | Competenza: {resp['score']:.2f}]\n{resp['response']}"
        )
    context = "\n\n---\n\n".join(parts)

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei l'orchestratore del sistema AGI Tree. Hai ricevuto risposte da "
                    "diversi modelli specializzati, ciascuno con un punteggio di competenza.\n"
                    "Sintetizza queste risposte in una risposta unica, coerente e completa.\n"
                    "Dai più peso alle risposte con score più alto.\n"
                    "Non menzionare il sistema interno di routing, gli score o i nomi degli esperti "
                    "nella risposta finale. Scrivi come se fossi tu a rispondere direttamente."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Domanda originale: {query}\n\n"
                    f"Risposte degli esperti:\n{context}\n\n"
                    "Sintetizza una risposta completa:"
                ),
            },
        ],
        temperature=0.7,
        max_tokens=3000,
    )
    return response.choices[0].message.content
