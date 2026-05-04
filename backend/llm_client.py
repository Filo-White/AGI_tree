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
# Phase 1: Chapter detection (regex + LLM fallback)
# ------------------------------------------------------------------

_CHAPTER_PATTERNS = [
    r'(?m)^[\s]*(?:CAPITOLO|Capitolo)\s+[\dIVXLCDM]+[.:)\-\s]*[^\n]+',
    r'(?m)^[\s]*(?:CAP\.)\s+[\dIVXLCDM]+[.:)\-\s]*[^\n]+',
    r'(?m)^[\s]*\d+[\.\)]\s+[A-ZÀÈÉÌÒÙ][^\n]{5,100}',
    r'(?m)^[\s]*[IVXLCDM]+[\.\)]\s+[A-ZÀÈÉÌÒÙ][^\n]{5,100}',
]


def regex_detect_chapters(text: str) -> list[dict]:
    for pattern in _CHAPTER_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if len(matches) >= 3:
            chapters = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                chapters.append({"name": m.group().strip(), "text": text[start:end].strip()})
            if chapters and matches[0].start() > 300:
                chapters.insert(0, {"name": "Introduzione", "text": text[:matches[0].start()].strip()})
            return chapters
    return []


async def detect_chapters(document_text: str, model: str = DEFAULT_MODEL) -> tuple[list[dict], str]:
    chapters = regex_detect_chapters(document_text)
    if chapters and len(chapters) >= 3:
        return chapters, "regex"

    truncated = document_text[:50000]
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Sei un analizzatore di struttura documentale. "
                "Identifica TUTTI i capitoli o sezioni principali del documento.\n"
                "Il documento DEVE essere coperto al 100%.\n\n"
                "Per ogni capitolo fornisci:\n"
                "- name: titolo del capitolo\n"
                "- start_text: le PRIME 8-12 PAROLE ESATTE dal testo originale\n\n"
                'Rispondi SOLO in JSON:\n'
                '{"chapters": [{"name": "Titolo", "start_text": "prime parole esatte..."}]}'
            )},
            {"role": "user", "content": f"Identifica tutti i capitoli:\n\n{truncated}"},
        ],
        temperature=0.1,
        max_completion_tokens=4000,
    )

    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        chapter_defs = json.loads(raw).get("chapters", [])
    except (json.JSONDecodeError, KeyError):
        return [{"name": "Documento completo", "text": document_text}], "fallback"

    if not chapter_defs:
        return [{"name": "Documento completo", "text": document_text}], "fallback"

    positions: list[tuple[int, str]] = []
    for ch in chapter_defs:
        st = ch.get("start_text", "")
        if not st:
            continue
        pos = document_text.find(st[:60])
        if pos == -1:
            words = st.split()[:6]
            if words:
                pat = r'\s+'.join(re.escape(w) for w in words)
                m = re.search(pat, document_text, re.IGNORECASE)
                pos = m.start() if m else -1
        if pos >= 0:
            positions.append((pos, ch["name"]))

    if not positions:
        return [{"name": "Documento completo", "text": document_text}], "fallback"

    positions.sort(key=lambda x: x[0])
    deduped = [positions[0]]
    for p in positions[1:]:
        if p[0] != deduped[-1][0]:
            deduped.append(p)
    positions = deduped

    chapters = []
    if positions[0][0] > 300:
        chapters.append({"name": "Introduzione", "text": document_text[:positions[0][0]].strip()})
    for i, (pos, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(document_text)
        ct = document_text[pos:end].strip()
        if ct:
            chapters.append({"name": name, "text": ct})

    return (chapters if chapters else [{"name": "Documento completo", "text": document_text}]), "llm"


# ------------------------------------------------------------------
# Phase 2: Section extraction per chapter
# ------------------------------------------------------------------

async def extract_sections_for_chapter(
    chapter_name: str, chapter_text: str, model: str = DEFAULT_MODEL
) -> list[dict]:
    truncated = chapter_text[:15000]
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Dato il testo di un capitolo, identifica TUTTE le sezioni o concetti distinti.\n"
                "Il capitolo DEVE essere coperto al 100%.\n\n"
                "Per ogni sezione:\n"
                "- name: nome breve (max 6 parole)\n"
                "- excerpt: ESTRATTO COMPLETO dal testo (300-800 caratteri)\n\n"
                "Identifica ALMENO 3 sezioni. Non saltare contenuto.\n\n"
                '{"sections": [{"name": "Nome", "excerpt": "testo..."}]}'
            )},
            {"role": "user", "content": f'Capitolo: "{chapter_name}"\n\nTesto:\n{truncated}'},
        ],
        temperature=0.2,
        max_completion_tokens=6000,
    )
    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        sections = json.loads(raw).get("sections", [])
        if sections:
            return sections
    except (json.JSONDecodeError, KeyError):
        pass
    return [{"name": chapter_name, "excerpt": chapter_text[:1000]}]


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


async def merge_chapter_lists(
    existing_names: list[str], new_names: list[str], model: str = DEFAULT_MODEL
) -> list[dict]:
    """Decide how to merge new chapters with existing ones."""
    ex_json = json.dumps(existing_names, ensure_ascii=False)
    nw_json = json.dumps(new_names, ensure_ascii=False)

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Devi decidere come integrare i capitoli di un nuovo documento con quelli esistenti.\n"
                "Per ogni capitolo nuovo, decidi se:\n"
                "- MERGE: corrisponde a uno esistente → indica quale\n"
                "- NEW: è un argomento nuovo → va aggiunto\n\n"
                '{"decisions": [{"new_chapter": "nome", "action": "merge"|"new", "merge_with": "nome o null"}]}'
            )},
            {"role": "user", "content": f"Esistenti: {ex_json}\nNuovi: {nw_json}"},
        ],
        temperature=0.1,
        max_completion_tokens=2000,
    )
    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        return json.loads(raw).get("decisions", [])
    except (json.JSONDecodeError, KeyError):
        return [{"new_chapter": n, "action": "new", "merge_with": None} for n in new_names]


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
