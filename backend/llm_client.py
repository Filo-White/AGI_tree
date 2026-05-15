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
# Document classification
# ------------------------------------------------------------------

async def classify_document(document_text: str, model: str = DEFAULT_MODEL) -> dict:
    """Classify document type and return splitting strategy."""
    preview = document_text[:5000]
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Classifica il tipo di documento e suggerisci come suddividerlo in nodi.\n\n"
                "Tipi possibili: book, paper, administrative, letter, manual, report, article, other\n\n"
                "Strategia di suddivisione:\n"
                "- book → 1 nodo per capitolo\n"
                "- paper → 1 nodo per sezione (Abstract, Introduction, Methods, Results, Discussion, Conclusion)\n"
                "- administrative → 1 nodo per articolo/clausola principale\n"
                "- letter → 1 nodo per argomento trattato\n"
                "- manual → 1 nodo per sezione/procedura\n"
                "- report → 1 nodo per sezione principale\n"
                "- article → 1 nodo per paragrafo tematico\n"
                "- other → 1 nodo per blocco tematico\n\n"
                'Rispondi SOLO in JSON:\n'
                '{"doc_type": "tipo", "description": "breve descrizione del documento", '
                '"split_hint": "come suddividere"}'
            )},
            {"role": "user", "content": f"Classifica questo documento:\n\n{preview}"},
        ],
        temperature=0.1,
        max_completion_tokens=300,
    )
    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        return json.loads(raw)
    except (json.JSONDecodeError, KeyError):
        return {"doc_type": "other", "description": "Documento generico", "split_hint": "blocchi tematici"}


# ------------------------------------------------------------------
# Top-level node detection (smart split based on doc type)
# ------------------------------------------------------------------

_CHAPTER_PATTERNS = [
    r'(?m)^[\s]*(?:CAPITOLO|Capitolo|CHAPTER|Chapter)\s+[\dIVXLCDM]+[.:)\-\s]*[^\n]+',
    r'(?m)^[\s]*(?:CAP\.)\s+[\dIVXLCDM]+[.:)\-\s]*[^\n]+',
    r'(?m)^[\s]*\d+[\.\)]\s+[A-ZÀÈÉÌÒÙ][^\n]{5,100}',
    r'(?m)^[\s]*[IVXLCDM]+[\.\)]\s+[A-ZÀÈÉÌÒÙ][^\n]{5,100}',
]


def regex_detect_sections(text: str) -> list[dict]:
    """Try to detect sections via regex patterns."""
    for pattern in _CHAPTER_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if len(matches) >= 2:
            sections = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append({"name": m.group().strip(), "text": text[start:end].strip()})
            if sections and matches[0].start() > 300:
                sections.insert(0, {"name": "Introduzione", "text": text[:matches[0].start()].strip()})
            return sections
    return []


async def detect_top_level_nodes(
    document_text: str, doc_type: str, model: str = DEFAULT_MODEL
) -> tuple[list[dict], str]:
    """Detect top-level nodes based on document type. Returns (nodes, method)."""
    # Try regex first
    sections = regex_detect_sections(document_text)
    if sections and len(sections) >= 2:
        return sections, "regex"

    # LLM-based detection
    truncated = document_text[:50000]

    type_instructions = {
        "book": "Identifica TUTTI i capitoli del libro.",
        "paper": "Identifica le sezioni principali del paper (Abstract, Introduction, Methods, Results, Discussion, Conclusion, ecc.).",
        "administrative": "Identifica gli articoli o clausole principali del documento.",
        "letter": "Identifica i diversi argomenti/temi trattati nella lettera.",
        "manual": "Identifica le sezioni o procedure principali del manuale.",
        "report": "Identifica le sezioni principali del report.",
        "article": "Identifica i paragrafi tematici principali dell'articolo.",
    }
    instruction = type_instructions.get(doc_type, "Identifica i blocchi tematici principali del documento.")

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                f"Sei un analizzatore di struttura documentale. Tipo documento: {doc_type}.\n"
                f"{instruction}\n"
                "Il documento DEVE essere coperto al 100%.\n"
                "Crea SOLO nodi di alto livello (NON sotto-sezioni).\n\n"
                "Per ogni nodo fornisci:\n"
                "- name: titolo della sezione\n"
                "- start_text: le PRIME 8-12 PAROLE ESATTE dal testo originale di quella sezione\n\n"
                'Rispondi SOLO in JSON:\n'
                '{"nodes": [{"name": "Titolo", "start_text": "prime parole esatte..."}]}'
            )},
            {"role": "user", "content": f"Identifica i nodi principali:\n\n{truncated}"},
        ],
        temperature=0.1,
        max_completion_tokens=4000,
    )

    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        node_defs = json.loads(raw).get("nodes", [])
    except (json.JSONDecodeError, KeyError):
        return [{"name": "Documento completo", "text": document_text}], "fallback"

    if not node_defs:
        return [{"name": "Documento completo", "text": document_text}], "fallback"

    # Locate positions in text
    positions: list[tuple[int, str]] = []
    for nd in node_defs:
        st = nd.get("start_text", "")
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
            positions.append((pos, nd["name"]))

    if not positions:
        return [{"name": "Documento completo", "text": document_text}], "fallback"

    positions.sort(key=lambda x: x[0])
    deduped = [positions[0]]
    for p in positions[1:]:
        if p[0] != deduped[-1][0]:
            deduped.append(p)
    positions = deduped

    nodes = []
    if positions[0][0] > 300:
        nodes.append({"name": "Introduzione", "text": document_text[:positions[0][0]].strip()})
    for i, (pos, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(document_text)
        ct = document_text[pos:end].strip()
        if ct:
            nodes.append({"name": name, "text": ct})

    return (nodes if nodes else [{"name": "Documento completo", "text": document_text}]), "llm"


# ------------------------------------------------------------------
# On-demand leaf expansion
# ------------------------------------------------------------------

async def expand_node_into_leaves(
    node_name: str, node_text: str, model: str = DEFAULT_MODEL
) -> list[dict]:
    """Expand a top-level node into sub-sections (leaves). Called on demand."""
    truncated = node_text[:15000]
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Dato il testo di una sezione, identifica TUTTE le sotto-sezioni o concetti distinti.\n"
                "La sezione DEVE essere coperta al 100%.\n\n"
                "Per ogni sotto-sezione:\n"
                "- name: nome breve (max 6 parole)\n"
                "- excerpt: ESTRATTO COMPLETO dal testo (300-800 caratteri)\n\n"
                "Identifica ALMENO 2 sotto-sezioni. Non saltare contenuto.\n\n"
                '{"leaves": [{"name": "Nome", "excerpt": "testo..."}]}'
            )},
            {"role": "user", "content": f'Sezione: "{node_name}"\n\nTesto:\n{truncated}'},
        ],
        temperature=0.2,
        max_completion_tokens=6000,
    )
    try:
        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removesuffix("```").strip()
        leaves = json.loads(raw).get("leaves", [])
        if leaves:
            return leaves
    except (json.JSONDecodeError, KeyError):
        pass
    return [{"name": node_name, "excerpt": node_text[:1000]}]


# ------------------------------------------------------------------
# Relevance check (before auto-expansion)
# ------------------------------------------------------------------

async def check_node_relevance(
    node_name: str, node_context: str, query: str, model: str = DEFAULT_MODEL
) -> bool:
    """Check if a node is actually relevant to the query (to avoid false auto-expansion)."""
    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Determina se la seguente sezione di un documento è PERTINENTE "
                "alla domanda dell'utente. La sezione non deve per forza rispondere completamente, "
                "ma l'argomento deve essere correlato.\n\n"
                "Rispondi SOLO con: SI o NO"
            )},
            {"role": "user", "content": (
                f"Sezione: {node_name}\n"
                f"Contesto: {node_context[:1000]}\n\n"
                f"Domanda: {query}\n\nÈ pertinente?"
            )},
        ],
        temperature=0.1,
        max_completion_tokens=5,
    )
    answer = response.choices[0].message.content.strip().upper()
    return "SI" in answer or "SÌ" in answer or "YES" in answer


# ------------------------------------------------------------------
# Node system prompt generation
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Merge logic for additional documents
# ------------------------------------------------------------------

async def merge_node_lists(
    existing_names: list[str], new_names: list[str], model: str = DEFAULT_MODEL
) -> list[dict]:
    """Decide how to merge new nodes with existing ones."""
    ex_json = json.dumps(existing_names, ensure_ascii=False)
    nw_json = json.dumps(new_names, ensure_ascii=False)

    response = await _get_client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Devi decidere come integrare le sezioni di un nuovo documento con quelle esistenti.\n"
                "Per ogni sezione nuova, decidi se:\n"
                "- MERGE: corrisponde a una esistente → indica quale\n"
                "- NEW: è un argomento nuovo → va aggiunto\n\n"
                '{"decisions": [{"new_node": "nome", "action": "merge"|"new", "merge_with": "nome o null"}]}'
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
        return [{"new_node": n, "action": "new", "merge_with": None} for n in new_names]


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
