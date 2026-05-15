# 🌳 AGI Tree

Sistema ad albero che costruisce **dinamicamente** una gerarchia di modelli LLM specializzati a partire dai documenti caricati dall'utente.

## Come funziona

1. **Carica un documento** (PDF, TXT, MD, CSV) tramite l'interfaccia web.
2. Il sistema analizza il documento in due fasi:
   - **Fase 1** — Rileva i capitoli (regex o LLM).
   - **Fase 2** — Per ogni capitolo estrae tutte le sezioni/concetti con estratti testuali.
3. Viene costruito un albero: **Root → Capitoli → Sezioni**.
4. Ogni nodo foglia (sezione) ha un proprio system prompt e il contesto estratto dal documento.
5. Quando fai una domanda, il sistema:
   - Valuta la competenza di ogni foglia (score 0–1)
   - Seleziona le foglie più competenti
   - Genera risposte specializzate
   - Sintetizza una risposta finale coerente

```
              [Orchestratore]
              /      |      \
      [Cap. 1]  [Cap. 2]  [Cap. 3]
       / | \      / | \      / | \
     sez sez sez sez sez sez sez sez sez
```

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, OpenAI API (`gpt-5.4-nano-2026-03-17`)
- **Frontend**: React 18, TypeScript, Vite, TailwindCSS, Lucide Icons
- **Comunicazione**: WebSocket per aggiornamenti in tempo reale

---

## Avvio rapido

### Prerequisiti

- **Python 3.11+** (consigliato: conda o venv)
- **Node.js 18+** e npm
- Una **API key OpenAI** con accesso al modello `gpt-5.4-nano-2026-03-17`

### 1. Clona il repository

```bash
git clone https://github.com/Filo-White/AGI_tree.git
cd AGI_tree
```

### 2. Configura la API key

```bash
cp .env.example .env
```

Apri `.env` e inserisci la tua chiave:

```
OPENAI_API_KEY=sk-...
```

### 3. Avvia il Backend

**Opzione A — Con conda (consigliato):**

```bash
conda create -n AGI_tree python=3.11 -y
conda activate AGI_tree
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Opzione B — Con venv:**

```bash
cd backend
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Il backend sarà attivo su `http://localhost:8000`.

### 4. Avvia il Frontend

In un **secondo terminale**:

```bash
cd frontend
npm install
npm run dev
```

Il frontend sarà attivo su `http://localhost:3000`.

### 5. Usa l'applicazione

1. Apri **http://localhost:3000** nel browser.
2. Clicca **"Carica documento"** e seleziona un PDF o TXT.
3. Attendi la costruzione dell'albero (vedrai il progresso in tempo reale nel tab **Log**).
4. Passa al tab **Analisi** per ispezionare come il documento è stato suddiviso.
5. Passa al tab **Albero** per vedere la struttura visiva con nodi circolari.
6. Scrivi una domanda nella chat a sinistra — il sistema instraderà la query ai nodi più competenti.

---

## Struttura del progetto

```
AGI_tree/
├── backend/
│   ├── main.py              # FastAPI app, endpoints, WebSocket
│   ├── llm_client.py        # Chiamate OpenAI (analisi, scoring, sintesi)
│   ├── tree_engine.py       # Logica albero (build, merge, query routing)
│   ├── document_processor.py # Estrazione testo da PDF/TXT
│   ├── models.py            # Modelli Pydantic
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Componente principale + AnalysisPanel
│   │   ├── types.ts         # Tipi TypeScript
│   │   └── components/
│   │       ├── TreeView.tsx  # Visualizzazione albero (nodi circolari)
│   │       ├── ChatPanel.tsx # Interfaccia chat
│   │       └── NodeDetail.tsx # Dettaglio nodo selezionato
│   ├── package.json
│   └── vite.config.ts
├── .env.example
└── README.md
```

## Features

- 📄 **Upload-first flow** — L'albero si genera automaticamente dal documento
- 🌳 **Visualizzazione ad albero** con nodi circolari e animazioni in tempo reale
- 🔍 **Pannello Analisi** — Ispeziona capitoli, sezioni e estratti assegnati a ogni nodo
- � **Scoring di competenza** — Ogni foglia si auto-valuta sulla domanda
- 💬 **Chat** — Risposte sintetizzate dai nodi più competenti
- 🔄 **Merge incrementale** — Carica più documenti, l'albero si arricchisce senza perdere dati
- 📡 **WebSocket** — Progresso live di ogni fase (analisi, building, scoring, risposta)
