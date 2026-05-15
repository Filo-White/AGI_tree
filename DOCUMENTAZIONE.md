# AGI — Documentazione Tecnica Completa

> "AGI is not an LLM (e fino a qui, siamo tutti d'accordo)"

---

> **Nota sul termine SLM:** Nel codice, il docstring di `router.py` riporta erroneamente "SLM (Semantic Language Map)". Il termine corretto, come confermato dall'autore e coerente con l'intera architettura del progetto (vedi `SLMAgent.py`), è **SLM = Small Language Model**.

---

## Indice

1. [Panoramica del Progetto](#1-panoramica-del-progetto)
2. [Architettura Generale](#2-architettura-generale)
3. [Concetto Fondamentale: SLM (Small Language Model)](#3-concetto-fondamentale-slm)
4. [Stack Tecnologico e Dipendenze](#4-stack-tecnologico-e-dipendenze)
5. [Struttura dei File](#5-struttura-dei-file)
6. [Fase 1 — Ingestion Pipeline](#6-fase-1--ingestion-pipeline)
7. [Fase 2 — Query Pipeline](#7-fase-2--query-pipeline)
8. [Storage e Persistenza](#8-storage-e-persistenza)
9. [Interfaccia Web — app.py](#9-interfaccia-web--apppy)
10. [Analisi Modulo per Modulo](#10-analisi-modulo-per-modulo)
11. [Costanti e Parametri Configurabili](#11-costanti-e-parametri-configurabili)
12. [Algoritmi Chiave](#12-algoritmi-chiave)
13. [Flusso Dati Completo](#13-flusso-dati-completo)
14. [Benchmark: RAG Standard vs SLM-RAG](#14-benchmark-rag-standard-vs-slm-rag)
15. [Visualizzazione 3D dello Spazio Vettoriale](#15-visualizzazione-3d-dello-spazio-vettoriale)

---

## 1. Panoramica del Progetto

Il progetto implementa un'architettura **multi-SLM (Small Language Model)** per il question answering su documenti. La tesi centrale — espressa già nel titolo e nel README — è che l'intelligenza non debba per forza emergere da un unico LLM grande: può emergere dalla **collaborazione di molti Small Language Model specializzati**, ognuno esperto in un dominio, coordinati da un sistema di routing semantico.

Il sistema è completamente locale: nessuna chiamata ad API esterne. Tutto gira on-premise con modelli HuggingFace.

### Tesi del progetto

Un RAG tradizionale usa un singolo grande modello che interroga un knowledge base monolitico. Questo progetto propone invece:

- Il knowledge base viene **partizionato** in domini tematici, ciascuno assegnato a un **Small Language Model** dedicato.
- Ogni SLM è un agente autonomo con un proprio corpus di chunk e, potenzialmente, il proprio modello linguistico per la generazione.
- Un **router semantico** seleziona quali SLM attivare per ogni query, sulla base della similarità tra la query e la firma semantica di ciascun SLM.
- Il risultato è un sistema più veloce (pool di ricerca ridotto), più preciso (contesto più focalizzato) e scalabile (aggiungere nuovi documenti crea nuovi SLM senza toccare quelli esistenti).

**Capacità implementate:**
- Ingestione di PDF di qualsiasi tipo con chunking adattivo al tipo di documento.
- Partizionamento automatico del corpus in SLM (per capitoli o per clustering coseno).
- Generazione automatica di topic summary e keywords per ogni SLM tramite Qwen2.5-7B.
- Routing semantico a query-time verso i top-N SLM più pertinenti.
- Retrieval ibrido (dense coseno + BM25 + RRF) sul pool ristretto degli SLM selezionati.
- Generazione streaming della risposta con qualsiasi modello HuggingFace.
- Visualizzazione interattiva 3D dello spazio vettoriale.
- Benchmark comparativo RAG standard vs SLM-routed.

---

## 2. Architettura Generale

Il sistema ha due livelli di implementazione:

**Livello concettuale completo** (`SLMAgent.py`): ogni SLM è un agente autonomo con il proprio modello linguistico caricabile, il proprio corpus e la propria logica RAG.

**Livello prototipo** (`app.py`): il routing e il retrieval multi-SLM sono pienamente funzionanti; la generazione usa un singolo modello condiviso (scelta pragmatica per la demo).

```
┌────────────────────────────────────────────────────────────────┐
│                      INGESTION PIPELINE                        │
│                                                                │
│  PDF ──► chunking.py ──► ChromaDB (chunk embeddings 768D)     │
│                │                                               │
│                └──► router.py ──► SLM Registry                │
│                          │         (uno entry per SLM)         │
│                          │                                     │
│                          └──► Qwen2.5-7B-Instruct             │
│                               topic_summary + keywords         │
│                               → summary_embedding (routing)   │
└────────────────────────────────────────────────────────────────┘
                               │
              registry.json  +  slm_data/*.json  +  chroma_db/
                               │
┌────────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                           │
│                                                                │
│  Query ──► all-mpnet-base-v2 ──► query_embedding              │
│                                        │                       │
│                        find_top_n_slms()                       │
│                        cosine(query, slm.summary_embedding)    │
│                                        │                       │
│                     top-N SLM selezionati                      │
│                                        │                       │
│                     _retrieve_hybrid()                         │
│                     dense cosine + BM25 + RRF                  │
│                     solo sul pool dei top-N SLM                │
│                                        │                       │
│  VISIONE COMPLETA (SLMAgent.py):       │                       │
│  ogni SLM usa il suo modello ──────────┤                       │
│  PROTOTIPO (app.py):                   │                       │
│  modello condiviso (user-specified) ───┘                       │
│                        risposta in streaming                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Concetto Fondamentale: SLM (Small Language Model)

Un **SLM (Small Language Model)** in questo progetto è un **agente linguistico specializzato** che combina:
- Un **corpus di chunk** tematicamente coerenti (la sua conoscenza di dominio).
- Una **firma semantica** (centroid + summary_embedding) che lo identifica per il routing.
- Un **modello linguistico dedicato** (`model_name`) per la generazione delle risposte sul suo dominio.

L'insieme di tutti gli SLM forma una **flotta di esperti complementari**: ognuno eccelle nel suo dominio, e il router seleziona il giusto esperto per ogni query.

> **Perché "Small"?** Ogni SLM può usare un modello leggero (es. 1-4B parametri) perché non deve essere generalista: il suo contesto di generazione è già filtrato e ristretto al suo corpus specializzato. Un modello piccolo su un dominio ristretto può superare un modello grande su un corpus generico.

### Struttura nel registry (identità dell'SLM)

```json
{
  "slm_a3f7b2c1": {
    "collection":         "nome_collection_chromadb",
    "chunks_json":        "slm_data/slm_a3f7b2c1_chunks.json",
    "chunk_count":        42,
    "centroid_embedding": [0.023, -0.157, "..."],
    "topic_summary":      "This section covers the economic causes of the French Revolution.",
    "keywords":           ["French Revolution", "taxation", "nobility", "..."],
    "summary_embedding":  [0.041, -0.203, "..."]
  }
}
```

### Struttura completa dell'agente (SLMAgent)

```python
@dataclass
class SLMAgent:
    name: str                                    # identificatore univoco
    model_name: str                              # es. "Qwen/Qwen2.5-1.5B-Instruct"
    chunks_json_path: str                        # corpus dell'SLM
    collection_name: str                         # collection ChromaDB
    centroid_embedding: Optional[List[float]]    # centro geometrico del dominio
    topic_summary: str                           # descrizione del dominio
    keywords: List[str]                          # termini chiave del dominio
    summary_embedding: Optional[List[float]]     # vettore di routing
    tokenizer: Optional[AutoTokenizer]           # runtime: tokenizer del modello
    model: Optional[AutoModelForCausalLM]        # runtime: modello caricato
    _chunk_ids: List[str]                        # cache locale degli ID chunk
    _last_modified: float                        # mtime per invalidazione cache
```

Una volta caricato con `load_slm_agent()`, l'SLM è autonomo: può eseguire l'intera pipeline RAG (`retrieve → build_prompt → generate`) con il suo modello senza dipendere da altri componenti.

### Doppia rappresentazione vettoriale

Ogni SLM ha due vettori con ruoli distinti:

- **`centroid_embedding`**: media normalizzata L2 degli embedding di tutti i chunk del corpus dell'SLM — usato per il clustering, l'assegnazione e il merge.
- **`summary_embedding`**: embedding delle keywords generate da Qwen — usato per il **routing a query-time**, perché cattura il significato tematico distillato da un LLM (più preciso della media grezza dei chunk).

### Campi chiave nel registry

| Campo | Tipo | Ruolo |
|---|---|---|
| `collection` | `str` | Collection ChromaDB con i chunk embedding del corpus |
| `chunks_json` | `str` | Path del file JSON con gli ID chunk del corpus |
| `chunk_count` | `int` | Dimensione del corpus |
| `centroid_embedding` | `List[float]` | Centro geometrico del dominio (768D, normalizzato) |
| `topic_summary` | `str` | Descrizione in una frase del dominio (generata da Qwen) |
| `keywords` | `List[str]` | 15-25 termini specifici in inglese (generati da Qwen) |
| `summary_embedding` | `List[float]` | Embedding delle keywords — **vettore di routing** |

---

## 4. Stack Tecnologico e Dipendenze

| Libreria | Versione | Ruolo |
|---|---|---|
| `gradio` | latest | Interfaccia web |
| `sentence-transformers` | latest | Embedding chunk e query (`all-mpnet-base-v2`, 768D) |
| `chromadb` | latest | Vector store persistente (spazio coseno) |
| `PyMuPDF` (`fitz`) | latest | Parsing PDF |
| `torch` | latest | Backend modelli HuggingFace |
| `transformers` | 4.32.1 | Tokenizer, causal LM, streaming |
| `numpy` | 1.24.3 | Operazioni vettoriali |
| `scikit-learn` | 1.3.0 | PCA, t-SNE |
| `langchain-text-splitters` | latest | `RecursiveCharacterTextSplitter` |
| `rank_bm25` | latest | BM25 per retrieval lessicale |
| `umap-learn` | opzionale | UMAP per visualizzazione |
| `hdbscan` | opzionale | Clustering alternativo batch |

**Modelli usati:**
- Embedding: `sentence-transformers/all-mpnet-base-v2`
- Summary/Keywords: `Qwen/Qwen2.5-7B-Instruct` (hardcoded, non configurabile)
- Generazione risposte: qualsiasi modello HuggingFace (default UI: `Qwen/Qwen3.5-4B`)

---

## 5. Struttura dei File

```
AGI/
├── app.py                  # Entrypoint principale — UI Gradio + orchestrazione
├── router.py               # Logica core SLM: creazione, assegnazione, merge, routing, summary
├── chunking.py             # Parsing PDF, rilevamento tipo, chunking adattivo, ChromaDB
├── SLMAgent.py             # Dataclass SLMAgent, prompt building, funzione RAG completa
├── benchmark.py            # Confronto RAG standard vs SLM-routed (metriche + Markdown)
├── extract_vectors.py      # Esportazione 3D con struttura SLM (PCA/t-SNE/UMAP)
├── extract_raw_vectors.py  # Esportazione 3D raw colorata per capitolo (no SLM)
├── visualizer.html         # Visualizzatore Three.js interattivo 3D
├── test_router.py          # Unit test logica router (ChromaDB in-memory, mock LLM)
├── test_retrieval.py       # CLI per testare retrieval su ChromaDB esistente
├── diagnose_slms.py        # Diagnostica similarità tra centroidi SLM
├── show_slms.py            # CLI per visualizzare summary e keywords degli SLM
├── debug.py                # Snippet di debug ChromaDB
├── requirements.txt        # Dipendenze Python
│
├── registry.json           # [generato] Registry JSON di tutti gli SLM
├── vectors.json            # [generato] Dati 3D per il visualizzatore
├── chroma_db/              # [generato] Database vettoriale ChromaDB persistente
└── slm_data/               # [generato] Un file JSON per SLM con gli ID dei chunk
    └── slm_<id>_chunks.json
```

---

## 6. Fase 1 — Ingestion Pipeline

L'ingestion è orchestrata da `app.py` → `upload_and_chunk()`, che chiama in sequenza:
`process_pdf()` → `assign_chunks_auto()` → `merge_close_slms()` → `refresh_all_summaries()`

### 6.1 Parsing e Rilevamento Tipo Documento

**`chunking.extract_chapters(pdf_path)`** — Legge il PDF con PyMuPDF e cerca intestazioni di capitoli via regex:
```
^(Chapter \d+...|CHAPTER \d+...|\d{1,2}.\d+ [A-Z]...|\d{1,2}. [A-Z]...)
```
Se non trova intestazioni, tratta il documento come un unico capitolo `"Document"`.

**`chunking.detect_doc_type(text)`** — Conta segnali euristici nei primi 8000 caratteri:

| Tipo | Segnali |
|---|---|
| `paper` | `abstract`, `introduction`, `doi`, `arxiv`, citazioni `[1]`, sezioni numerate |
| `book` | `chapter`, `preface`, molti `\n\n`, documento >50k caratteri |
| `technical` | `api`, `endpoint`, `def`, `class`, backtick code, `parameter`, `installation` |
| `generic` | fallback se nessun tipo supera 2 segnali |

### 6.2 Chunking Adattivo

**`chunking.get_chunker(doc_type)`** — Configura `RecursiveCharacterTextSplitter`:

| Tipo | `chunk_size` | `chunk_overlap` |
|---|---|---|
| `paper` | 800 | 100 |
| `book` | 2000 | 150 |
| `technical` | 1200 | 200 |
| `generic` | 1500 | 150 |

Separatori in ordine: `"\n\n"`, `"\n"`, `"."`, `" "`, `""`.

Un filtro regex rimuove chunk di rumore (indici, glossari, numeri di pagina).

### 6.3 Embedding e Salvataggio in ChromaDB

**`chunking.process_pdf()`** — Batch embedding con `all-mpnet-base-v2` (batch_size=32, normalizzati). Ogni chunk salvato in ChromaDB con `id` UUID, vettore 768D, testo e `metadata: {"chapter": "..."}`. La collection usa spazio coseno (`"hnsw:space": "cosine"`).

### 6.4 Assegnazione dei Chunk agli SLM

**`router.assign_chunks_auto()`** — Sceglie automaticamente la strategia:

- **Strategia `"chapter"`** (se esistono più capitoli distinti): `assign_chunks_by_chapter()` → un SLM per capitolo, centroide = media degli embedding del capitolo normalizzata.
- **Strategia `"cosine"`** (documento senza capitoli): `assign_chunks()` → per ogni chunk, trova l'SLM con centroide più simile (soglia=0.55); se nessuno supera la soglia, crea un nuovo SLM. Il centroide viene aggiornato **incrementalmente** (vedi §12).

### 6.5 Merge degli SLM

**`router.merge_close_slms(threshold=0.88)`** — Algoritmo iterativo:
1. Calcola cosine similarity tra tutti i centroidi a coppie.
2. Se la coppia con score massimo supera la soglia: unisce i due SLM.
3. Mantiene l'SLM con più chunk, calcola **centroide pesato**: `(c_a*n_a + c_b*n_b)/(n_a+n_b)` normalizzato.
4. Ripete fino a convergenza.

### 6.6 Generazione Summary con Qwen

**`router.refresh_all_summaries()`** — Dopo ogni ingestion, esegue Qwen2.5-7B-Instruct su ogni SLM.

**Selezione chunk rappresentativi** (`_select_representative_chunks`):
- Top-20 chunk con similarità coseno più alta al centroide (nucleo tematico).
- Top-5 chunk con similarità più bassa (diversità periferica).
- Massimo 10 passati a Qwen, troncati a 400 caratteri ciascuno.

**Prompt a Qwen:**
```
You are a knowledge indexing assistant. Given the following text chunks, extract:
1. A SUMMARY: one sentence describing the main topic.
2. KEYWORDS: 15-25 specific keywords (comma-separated, always in ENGLISH).
   Include: named entities, technical terms, places, people, events, concepts.
   Do NOT include generic words like 'chapter', 'text', 'document'.
```

**Post-processing:** keywords pulite da rumore (>60 caratteri o che iniziano con pattern meta-commentary). Il `summary_embedding` viene calcolato embeddinando le **keywords** (non il testo del summary) — segnale semantico più denso e preciso per il routing.

---

## 7. Fase 2 — Query Pipeline

### 7.1 Routing Semantico

**`router.find_top_n_slms(query_emb, registry, n=3)`**

Per ogni SLM con `summary_embedding` valido, calcola la cosine similarity tra il vettore query e il `summary_embedding`. Restituisce i top-N ordinati per score decrescente. SLM senza summary_embedding vengono ignorati.

> Il routing usa `summary_embedding` (keywords di Qwen), **non** il `centroid_embedding`. Le keyword catturano il significato tematico distillato da un LLM, allineandosi meglio all'intento della query.

### 7.2 Retrieval Ibrido Dense + BM25 + RRF

**`app._retrieve_hybrid(query, query_emb, top_slms, registry, top_k=5)`**

Per ogni SLM selezionato, recupera embedding e testi da ChromaDB. Poi, sull'insieme unificato dei chunk dei top-N SLM:

1. **Dense ranking**: chunk ordinati per similarità coseno con la query.
2. **BM25 ranking**: `BM25Okapi` su corpus tokenizzato (lowercase, rimozione punteggiatura).
3. **RRF fusion**: $\text{score}(d) = \sum_r \frac{1}{60 + \text{rank}_r(d)}$, top-K restituiti.

### 7.3 Generazione della Risposta

**`app._generate_streaming(query, chunks, model_name, max_tokens)`** (opzionale)

Questa è la funzione di generazione del **prototipo**: usa un singolo modello condiviso scelto dall'utente, non il modello specifico dell'SLM selezionato. È una scelta pragmatica per la demo (caricare N modelli diversi simultaneamente è costoso).

Nella visione completa dell'architettura (implementata in `SLMAgent.py`), ogni SLM usa **il proprio modello** (`agent.model_name`) per generare la risposta sul suo dominio tramite la funzione `rag()`.

**Parametri del prototipo:**
- Modello caricato da HuggingFace con cache in `_model_cache` (reload evitato tra query successive).
- Prompt costruito da `SLMAgent.build_prompt()`: istruzioni di grounding + contesto numerato + domanda.
- Generazione in **streaming** via `TextIteratorStreamer` (thread separato per non bloccare l'UI).
- Tag `<think>...</think>` rimossi in real-time via regex (per modelli reasoning come Qwen3/DeepSeek-R1).
- `max_new_tokens=256`, `temperature=0.1`, `do_sample=False` (greedy).

---

## 8. Storage e Persistenza

| Storage | Path | Contenuto |
|---|---|---|
| ChromaDB | `./chroma_db/` | Embeddings 768D, testi, metadata dei chunk |
| Registry | `./registry.json` | Dizionario `{slm_name: entry}` con centroidi, summary, keywords |
| SLM data | `./slm_data/<slm>_chunks.json` | Lista `[{"id": "uuid"}, ...]` per ogni SLM |
| HDBSCAN model | `./hdbscan_model.pkl` | Modello fittato + PCA (solo se usato clustering alternativo) |

Il design separa l'indice leggero (chi appartiene a chi, in JSON) dallo storage pesante (embedding in ChromaDB), minimizzando I/O durante il routing.

---

## 9. Interfaccia Web — app.py

Costruita con **Gradio**, tema scuro custom (CSS con variabili, font Inter + JetBrains Mono, palette viola `#6366f1`).

### Tab "Ingestion"

| Componente | Funzione |
|---|---|
| File upload PDF | Carica il documento |
| Nome collection | Identifica la collection ChromaDB |
| Pulsante "Avvia Chunking" | Lancia l'intera pipeline di ingestion |
| Area "Stato" | Risultato: n. chunk, tipo documento, SLM aggiornati, merge effettuati, summary generati |
| Area "Anteprima" | JSON dei primi 5 chunk (id, chapter, testo preview) |
| Dropdown "Collection" | Lista aggiornata delle collection in ChromaDB |
| Area "Registry SLM" | Stato del registry: nome, chunk count, collection, preview summary |

### Tab "Query"

| Componente | Funzione |
|---|---|
| Textbox "Domanda" | Query in linguaggio naturale |
| Textbox "Modello HuggingFace" | Modello generativo (default `Qwen/Qwen3.5-4B`; se vuoto → solo retrieval) |
| Pulsante "Invia Query" | Lancia il pipeline di query |
| Area "Routing" | SLM selezionati con score, n. chunk, pool totale, device |
| Area "Risposta" | Risposta in streaming (o messaggio solo-retrieval) |
| Area "Chunk recuperati" | Top-K chunk con score coseno, SLM sorgente, capitolo, testo (max 420 char) |

---

## 10. Analisi Modulo per Modulo

### router.py (926 righe)

Modulo centrale. Gestisce l'intero ciclo di vita degli SLM.

**Funzioni principali:**

| Funzione | Scopo |
|---|---|
| `load_registry()` / `save_registry()` | I/O JSON del registry |
| `_update_centroid(old, new_emb, n)` | Aggiornamento incrementale centroide normalizzato O(1) |
| `_select_representative_chunks()` | Top-centrale + top-periferici per input a Qwen |
| `_qwen_summary_fn(texts)` | Genera topic_summary e keywords con Qwen2.5-7B (cache modulo) |
| `update_slm_summary()` | Aggiorna summary di un singolo SLM |
| `refresh_all_summaries()` | Rigenera summary per tutti gli SLM |
| `create_slm()` | Crea nuovo SLM nel registry |
| `assign_chunk()` | Assegna singolo chunk (cosine threshold o crea nuovo SLM) |
| `assign_chunks_by_chapter()` | Strategia capitoli: 1 SLM per capitolo |
| `assign_chunks_auto()` | Entry point: sceglie strategia automaticamente |
| `cluster_chunks_hdbscan()` | Clustering alternativo HDBSCAN + PCA (opzionale) |
| `find_top_n_slms()` | Routing query-time via cosine su summary_embedding |
| `merge_close_slms()` | Merge iterativo SLM con centroidi troppo simili |
| `migrate_centroids_to_summaries()` | Migrazione dati legacy (da TF-IDF a embedding) |
| `unload_summary_model()` | Dealloca Qwen da memoria (gc + CUDA empty_cache) |

**Cache Qwen:** variabili globali `_summary_tok` e `_summary_mdl` — modello caricato una volta e riusato per tutta la sessione.

---

### chunking.py

| Funzione | Scopo |
|---|---|
| `detect_doc_type(text)` | Heuristica conteggio segnali → tipo documento |
| `get_chunker(doc_type)` | Restituisce `RecursiveCharacterTextSplitter` configurato |
| `extract_chapters(pdf_path)` | PyMuPDF + regex → lista `[{chapter, text}]` |
| `process_pdf(...)` | Pipeline completa: parsing → chunking → embedding → ChromaDB |

---

### SLMAgent.py

È il modulo che realizza la **visione completa** del concetto SLM (Small Language Model). Definisce la dataclass `SLMAgent` che unifica in un unico oggetto:
- L'**identità del dominio**: `centroid_embedding`, `topic_summary`, `keywords`, `summary_embedding`.
- Il **corpus specializzato**: `chunks_json_path`, `collection_name`.
- Il **modello linguistico dedicato**: `model_name`, `tokenizer`, `model` (caricati a runtime).
- Una **cache locale** degli ID chunk con invalidazione automatica via `mtime`.

Una volta caricato, ogni SLM è completamente autonomo: esegue l'intera pipeline RAG internamente con il proprio modello, senza dipendere da altri componenti.

| Funzione | Scopo |
|---|---|
| `load_slm_agent(agent, device)` | Carica tokenizer + `AutoModelForCausalLM` dal `model_name` dell'SLM |
| `_sync_chunk_ids(agent)` | Ricarica chunk IDs da disco solo se il file è cambiato (via `mtime`) |
| `get_agent_chunk_ids(agent)` | Restituisce gli ID del corpus aggiornati |
| `retrieve_top_k(query, chunk_ids, collection, emb_model, top_k)` | Retrieval dense-only via ChromaDB `.query()` limitato al corpus dell'SLM |
| `build_prompt(query, chunks)` | Costruisce il prompt con istruzioni di grounding stricto: rispondere solo dal contesto |
| `generate_answer(prompt, tokenizer, model, ...)` | Generazione testo con il modello dell'SLM |
| `rag(query, agent, chroma_client, emb_model, ...)` | **Pipeline RAG completa** dell'SLM: retrieve → build_prompt → generate → dict risultato |

Il `app.py` usa solo il routing e il retrieval multi-SLM, ma delega la generazione a un singolo modello condiviso (scelta pragmatica del prototipo). `SLMAgent.py` è il codice che mostra come il sistema è pensato per funzionare nella versione completa: ogni SLM genera con il **suo** modello.

---

### benchmark.py

Confronto su 10 query storiche (inglese + italiano). Per ogni query:
- `retrieve_standard()`: full-corpus dense + BM25 + RRF.
- `retrieve_slm()`: router → top-3 SLM → pool ridotto → dense + BM25 + RRF.
- Generazione risposta con modello configurabile.

**Metriche:** latenza (ms), speedup, pool size, keyword hit rate, overlap@5. Output: `benchmark_answers.md`.

---

### extract_vectors.py

Esporta `vectors.json` per il visualizzatore 3D con struttura SLM:
1. Carica registry e mappa chunk→SLM.
2. Carica embedding da ChromaDB.
3. Riduce a 3D (chunk + centroidi insieme): **PCA**, **t-SNE**, **UMAP**.
4. Normalizza in `[-1, 1]`.
5. Calcola raggi per ogni centroide (max/mean/std distanza dai chunk assegnati nel 3D).

---

### extract_raw_vectors.py

Come `extract_vectors.py` ma senza struttura SLM. Chunk colorati per **capitolo**. Utile per esplorare la distribuzione naturale prima di decidere come clusterizzare. `centroid_info` vuoto nell'output.

---

### visualizer.html

File HTML/JS standalone con **Three.js**:
- Carica `vectors.json` da file picker locale (no server).
- Seleziona metodo di riduzione (PCA/t-SNE/UMAP).
- Punti colorati per SLM, sfere semitrasparenti intorno ai centroidi.
- Tooltip hover con preview chunk, SLM, capitolo.
- Controlli orbita/zoom/pan.

---

### test_router.py

Suite unit test senza PDF né LLM reali (ChromaDB in-memory, mock `summary_fn`):

| Test | Verifica |
|---|---|
| `test_incremental_centroid()` | Centroide incrementale ≈ media batch (cosine > 0.94); normalizzato |
| `test_registry_structure()` | Tutti i campi obbligatori presenti con tipi corretti |
| `test_representative_chunks()` | Numero chunk selezionati ragionevole (≥1, ≤totale) |
| `test_routing_score()` | Due gruppi semantici distinti → routing corretto; score = cosine esatta |
| `test_merge()` | Merge riduce o mantiene il numero di SLM |

Registry backuppato prima e ripristinato dopo ogni test.

---

### test_retrieval.py

CLI: `python test_retrieval.py "domanda" --top_k 10`. Retrieval su ChromaDB esistente, converte distanze L2 → cosine similarity (`1 - d²/2`). Mostra score, SLM, capitolo, preview testo.

---

### diagnose_slms.py

Read-only. Calcola tutte le similarità coseno pairwise tra centroidi. Mostra distribuzione (max/mean/median/min) e stima quanti SLM resterebbero a ogni soglia di merge — utile per calibrare `threshold` prima di eseguire il merge.

---

### show_slms.py

CLI: `python show_slms.py [--sort chunks|name] [--min-chunks N]`. Stampa summary e keywords di ogni SLM, con deduplicazione keywords case-insensitive.

---

### debug.py

Snippet minimale per ispezionare una collection ChromaDB: lista collection, conta chunk, interroga per metadati.

---

## 11. Costanti e Parametri Configurabili

### app.py

| Costante | Valore | Descrizione |
|---|---|---|
| `TOP_N_SLMS` | 3 | SLM selezionati dal router per ogni query |
| `TOP_K_CHUNKS` | 5 | Chunk restituiti dopo RRF |
| `MAX_TOKENS` | 256 | Token massimi generati |
| `RRF_K` | 60 | Parametro k nella formula RRF |

### router.py

| Costante | Valore | Descrizione |
|---|---|---|
| `SLM_SUMMARY_EVERY_N` | 20 | Refresh summary ogni N chunk durante ingestion |
| `TOP_REPRESENTATIVE` | 20 | Chunk centrali passati a Qwen |
| `TOP_DIVERSE` | 5 | Chunk periferici passati a Qwen |
| `SUMMARY_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | Modello fisso per summary |

### Soglie

| Parametro | Valore | Descrizione |
|---|---|---|
| `threshold` in `assign_chunks_auto` | 0.55 | Soglia coseno per assegnare chunk a SLM esistente |
| `threshold` in `merge_close_slms` | 0.88 | Soglia coseno sopra cui due SLM vengono uniti |
| `noise_threshold` in HDBSCAN | 0.60 | Frazione massima noise prima di refitting |

---

## 12. Algoritmi Chiave

### Centroide Incrementale

Aggiornamento in O(1) senza ricalcolare la media su tutti i chunk:

```
centroid_new = (centroid_old * (n-1) + new_embedding) / n
centroid_new = centroid_new / ||centroid_new||   # normalizzazione L2
```

La normalizzazione ad ogni passo introduce una piccola approssimazione rispetto alla media batch esatta, ma la cosine similarity con la media batch supera 0.94 (verificato dai test).

### Reciprocal Rank Fusion (RRF)

$$\text{score}_{RRF}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}_r(d)}$$

Con `k=60`. Assegna punteggi più alti ai chunk in cima a entrambi i ranking (dense e BM25). Robusto a scale diverse dei due sistemi di scoring.

### Selezione Chunk Rappresentativi

```
scored = [(i, cosine(chunk_i, centroid)) for i in chunk_ids]
scored.sort(descending)
selected = scored[:TOP_REPRESENTATIVE]    # nucleo semantico
selected ∪= scored[-TOP_DIVERSE:]         # diversità periferica
```

I chunk periferici evitano che Qwen generi summary troppo ristretti ignorando sottotemi.

### Clustering HDBSCAN (alternativo, non nel flusso principale)

Funzione `cluster_chunks_hdbscan()`:
1. Recupera tutti gli embedding dalla collection.
2. **PCA** 768D→50D (le differenze di densità in alta dimensionalità collassano senza riduzione).
3. **HDBSCAN** con metrica euclidea (corretta dopo PCA).
4. Noise points (label=-1) riassegnati al cluster più vicino per cosine.
5. Salva `hdbscan_model.pkl`; per ingestioni successive usa `approximate_predict()`.
6. Se noise fraction > `noise_threshold`, refitting completo da zero.

---

## 13. Flusso Dati Completo

```
PDF
 │
 ├─► extract_chapters() ──────────────────► [{chapter, text}, ...]
 ├─► detect_doc_type()  ──────────────────► "paper"|"book"|"technical"|"generic"
 ├─► RecursiveCharacterTextSplitter ──────► raw text chunks
 ├─► all-mpnet-base-v2.encode() ──────────► embeddings[N × 768]
 └─► ChromaDB.add() ──────────────────────► {id, embedding, document, metadata}

 ──► assign_chunks_auto()
      ├── capitoli > 1 ──► assign_chunks_by_chapter()  (1 SLM/capitolo)
      └── 1 capitolo   ──► assign_chunks()              (cosine clustering)

 ──► merge_close_slms(threshold=0.88)
      └── centroide pesato + normalizzazione

 ──► refresh_all_summaries()
      ├── _select_representative_chunks()  (top-central + top-diverse)
      ├── Qwen2.5-7B-Instruct              (topic_summary + keywords)
      └── all-mpnet-base-v2.encode(keywords) ──► summary_embedding

 ──── registry.json aggiornato ────────────────────────────────────


QUERY:
 query_string
  ├─► all-mpnet-base-v2.encode() ──────────► query_emb [768]
  ├─► find_top_n_slms()          ──────────► [(slm_1, 0.87), (slm_2, 0.74), ...]
  ├─► _retrieve_hybrid()
  │    ├── cosine scores ─────────────────── dense_ranking
  │    ├── BM25Okapi     ─────────────────── bm25_ranking
  │    └── RRF([dense, bm25]) ─────────────► top-K chunks
  ├─► build_prompt(query, chunks) ─────────► prompt
  └─► LLM.generate() streaming ────────────► risposta (think-tags rimossi)
```

---

## 14. Benchmark: RAG Standard vs SLM-RAG

Il benchmark (`benchmark.py`) confronta su 10 query storiche:

| | StdRAG | SLM-RAG |
|---|---|---|
| **Pool di ricerca** | Tutti i chunk della collection | Solo chunk dei top-3 SLM |
| **Retrieval** | Dense + BM25 + RRF su full corpus | Dense + BM25 + RRF su pool ridotto |
| **Overhead aggiuntivo** | — | Routing (cosine su N summary_embedding) |

**Vantaggi attesi di SLM-RAG:**
- **Speedup retrieval**: pool ridotto → meno operazioni vettoriali e BM25.
- **Keyword hit rate** uguale o superiore: routing riduce il rumore semantico.
- **Overhead routing trascurabile**: solo N prodotti scalari tra vettori 768D.

Output: `benchmark_answers.md` con tabella aggregata (speedup, pool reduction, keyword hit, overlap@5) e per ogni query le risposte comparative dei due sistemi.

---

## 15. Visualizzazione 3D dello Spazio Vettoriale

### Due modalità di esportazione

**`extract_vectors.py`** (modalità SLM):
```bash
python extract_vectors.py --method pca
python extract_vectors.py --method tsne --perplexity 40
python extract_vectors.py --method all
```
Chunk colorati per SLM, centroidi come sfere con raggi calcolati. Permette di vedere la coesione e la separazione dei cluster semantici.

**`extract_raw_vectors.py`** (modalità raw):
```bash
python extract_raw_vectors.py --method pca
python extract_raw_vectors.py --collection LibroStoria
```
Chunk colorati per capitolo, senza struttura SLM. Utile per esplorare la distribuzione naturale degli embedding prima di clusterizzare.

### Riduzione dimensionale supportata

| Metodo | Caratteristiche |
|---|---|
| **PCA** | Lineare, deterministico, veloce. Mostra la varianza principale. |
| **t-SNE** | Non-lineare, preserva struttura locale, lento su corpus grandi. |
| **UMAP** | Non-lineare, preserva struttura globale e locale, richiede `umap-learn`. |

Tutte le coordinate vengono normalizzate in `[-1, 1]` prima dell'esportazione.

### Struttura `vectors.json`

```json
{
  "meta": {
    "n_chunks": 1234, "n_slms": 18,
    "original_dim": 768, "methods": ["pca"]
  },
  "chunk_meta": [
    {"collection": "...", "slm": "slm_xxx", "id": "...", "chapter": "...", "preview": "..."}
  ],
  "centroid_info": [
    {"slm_name": "slm_xxx", "chunk_count": 42, "topic_summary": "...", "keywords": [...]}
  ],
  "reductions": {
    "pca": {
      "chunks":    [[x, y, z], ...],
      "centroids": [[x, y, z], ...],
      "radii":     [{"max_radius": 0.45, "mean_radius": 0.21, "std_radius": 0.08}]
    }
  }
}
```

Il file `visualizer.html` carica questo JSON tramite file picker locale (nessun server necessario) e renderizza la scena 3D interattiva con Three.js.
