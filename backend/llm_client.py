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


# ------------------------------------------------------------------
# Document analysis — extract topics and subtopics
# ------------------------------------------------------------------

async def analyze_document_topics(document_text: str, model: str = DEFAULT_MODEL) -> list[dict]:
    """Analyze a document and return main topics with subtopics and excerpts."""
    truncated = document_text[:30000]

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei un analizzatore di documenti. Analizza il testo fornito e identifica "
                    "i MACRO-ARGOMENTI principali trattati (es: Storia, Fisica, Economia, Medicina...).\n"
                    "Per ogni macro-argomento, identifica i SOTTO-ARGOMENTI specifici presenti nel documento "
                    "(es: sotto Storia potresti trovare 'Seconda Guerra Mondiale', 'Medioevo', ecc.).\n"
                    "Per ogni sotto-argomento, estrai un breve estratto rilevante dal testo (max 500 caratteri).\n\n"
                    "Rispondi SOLO in formato JSON:\n"
                    '{"topics": [\n'
                    '  {"name": "Nome Macro-Argomento", "subtopics": [\n'
                    '    {"name": "Nome Sotto-Argomento", "excerpt": "estratto rilevante dal testo..."}\n'
                    "  ]}\n"
                    "]}"
                ),
            },
            {"role": "user", "content": f"Analizza questo documento:\n\n{truncated}"},
        ],
        temperature=0.2,
        max_completion_tokens=4000,
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.removeprefix("```json").removesuffix("```").strip()
        result = json.loads(text)
        return result.get("topics", [])
    except (json.JSONDecodeError, KeyError):
        return []


async def generate_node_system_prompt(
    role_description: str, model: str = DEFAULT_MODEL
) -> str:
    """Generate a specialized system prompt for a tree node."""
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Genera un system prompt dettagliato per un modello LLM specializzato.\n"
                    "Il prompt deve descrivere la competenza specifica, il tono da usare, "
                    "e come l'esperto deve rispondere alle domande nel suo ambito.\n"
                    "Rispondi SOLO con il testo del system prompt, nient'altro."
                ),
            },
            {
                "role": "user",
                "content": f"Genera il system prompt per un esperto di: {role_description}",
            },
        ],
        temperature=0.4,
        max_completion_tokens=500,
    )
    return response.choices[0].message.content.strip()


async def merge_document_into_topics(
    existing_topics: list[dict], new_document_text: str, model: str = DEFAULT_MODEL
) -> list[dict]:
    """Analyze a new document and merge its topics with existing ones."""
    truncated = new_document_text[:30000]
    existing_json = json.dumps(existing_topics, ensure_ascii=False, indent=2)

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Sei un analizzatore di documenti. Devi analizzare un NUOVO documento "
                    "e integrare i suoi argomenti con quelli GIA' ESISTENTI.\n\n"
                    "Regole:\n"
                    "- Se un argomento del nuovo documento corrisponde a uno esistente, "
                    "ARRICCHISCI i sotto-argomenti e gli estratti di quello esistente.\n"
                    "- Se il nuovo documento contiene argomenti nuovi, AGGIUNGILI.\n"
                    "- Non eliminare mai argomenti o sotto-argomenti esistenti.\n"
                    "- Per sotto-argomenti esistenti arricchiti, concatena i nuovi estratti "
                    "con quelli vecchi.\n\n"
                    "Rispondi SOLO in formato JSON:\n"
                    '{"topics": [\n'
                    '  {"name": "Nome Macro-Argomento", "subtopics": [\n'
                    '    {"name": "Nome Sotto-Argomento", "excerpt": "estratto completo..."}\n'
                    "  ]}\n"
                    "]}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Argomenti esistenti:\n{existing_json}\n\n"
                    f"Nuovo documento da integrare:\n{truncated}"
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=5000,
    )
    try:
        text = response.choices[0].message.content.strip()
        text = text.removeprefix("```json").removesuffix("```").strip()
        result = json.loads(text)
        return result.get("topics", existing_topics)
    except (json.JSONDecodeError, KeyError):
        return existing_topics


# ------------------------------------------------------------------
# Query processing — scoring, answering, decomposition, synthesis
# ------------------------------------------------------------------

async def get_competence_score(system_prompt: str, query: str, context: str = "", model: str = DEFAULT_MODEL) -> float:
    messages = [{"role": "system", "content": system_prompt}]
    prompt = (
        "Valuta la tua competenza sulla seguente domanda. "
        "Rispondi SOLO con un numero decimale tra 0.0 e 1.0, dove "
        "0.0 = nessuna competenza e 1.0 = massima competenza.\n\n"
    )
    if context:
        prompt += f"Contesto disponibile:\n{context[:2000]}\n\n"
    prompt += f"Domanda: {query}\n\nScore:"

    messages.append({"role": "user", "content": prompt})

    response = await _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_completion_tokens=10,
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
    context: str = "",
    model: str = DEFAULT_MODEL,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]

    if context:
        user_content = (
            f"Contesto dal documento:\n---\n{context}\n---\n\n"
            f"Domanda: {query}"
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})

    response = await _get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=2000,
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
        max_completion_tokens=500,
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
        max_completion_tokens=3000,
    )
    return response.choices[0].message.content
