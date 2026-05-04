# 🌳 AGI Tree

A tree-based LLM routing system that distributes queries across a hierarchy of specialized AI models.

## Architecture

```
            [Root - Orchestrator]
               /            \
        [Node A]           [Node B]
        /      \           /      \
   [Leaf 1] [Leaf 2] [Leaf 3] [Leaf 4]
```

Each node is an LLM. Leaves are domain specialists. The system operates in **3 phases**:

1. **Discovery (top-down)** — The root propagates the query down the tree. Each leaf self-scores its competence (0–1).
2. **Score bubble-up (bottom-up)** — Scores flow back to the root, which now knows which leaves are most competent.
3. **Routing & Response** — The root decomposes the query if needed, routes sub-queries to the best leaves, collects answers, and synthesizes a final response via weighted merge.

## Tech Stack

- **Backend**: Python, FastAPI, OpenAI API (`gpt-5.4-nano`)
- **Frontend**: React, TypeScript, Vite, TailwindCSS
- **Communication**: WebSocket for real-time progress updates

## Setup

### 1. Clone & configure

```bash
git clone https://github.com/Filo-White/AGI_tree.git
cd AGI_tree
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Features

- 💬 Chat interface with the root orchestrator
- 🌳 Live animated tree visualization showing query routing in real-time
- 📊 Competence scoring and leaf selection
- 📄 Document upload (PDF, TXT) for context-aware answers
- ⚙️ Configurable tree structure via JSON editor
- 🔍 Node detail inspector (model, role, system prompt, scores)
