# Document Agent Tree (DAT)

**Agentic document QA via structure-aware hierarchical routing.**

DAT dynamically builds a tree of specialized LLM agents from an uploaded document's structure, then routes queries to the most relevant node using self-scoring and lazy expansion. It outperforms flat RAG, section-level RAG, and long-context baselines on answer quality while providing interpretable evidence localization.

---

## Key Results

| Method | Correctness | Faithfulness | Hit@3 | Abstention Acc. |
|--------|:-----------:|:------------:|:-----:|:---------------:|
| Flat RAG | 3.26 | 3.57 | 0.762 | 0.40 |
| Section-RAG | 3.48 | 3.48 | 0.951 | 0.53 |
| Long-Context | 3.85 | 4.05 | N/A | 0.67 |
| **DAT (ours)** | **4.16** | **4.23** | **0.992** | **0.80** |

> Evaluated on 15 real-world documents (122 queries) with dual LLM judges. Bootstrap 95% CIs confirm non-overlapping intervals between DAT and all baselines on correctness. See [`experiments/evaluation/`](experiments/evaluation/) for full methodology and reproducibility details.

---

## How It Works

```
                 ┌──────────────┐
   Document ───▶ │  Classifier  │───▶ doc_type (paper, regulation, manual, ...)
                 └──────────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  Decomposer  │───▶ Top-level nodes (chapters, sections, titles)
                 └──────────────┘
                        │
                        ▼
         ┌──────┬───────┼───────┬──────┐
         │      │       │       │      │
       Node₁  Node₂  Node₃  Node₄  Node₅   ← each holds a text segment
         │
         ▼ (lazy expansion if no node scores ≥ τ)
    ┌────┼────┐
  Leaf₁ Leaf₂ Leaf₃
```

1. **Classify** — An LLM identifies the document type (book, paper, regulation, manual, report, etc.)
2. **Decompose** — Top-level sections are extracted based on the document type; each becomes a node with its own context.
3. **Score** — Given a query, every node self-scores its relevance (0–1).
4. **Expand (lazy)** — If no node scores above threshold τ = 0.7 but the best node is query-relevant, it expands into sub-sections (leaves) and re-scores.
5. **Answer** — The highest-scoring node/leaf generates the final answer with its context window.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.11+, FastAPI, WebSocket |
| LLM | OpenAI API (`gpt-5.4-nano-2026-03-17`) |
| Frontend | React 18, TypeScript, Vite, TailwindCSS |
| Document parsing | PyPDF2, plain text |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- OpenAI API key with access to `gpt-5.4-nano-2026-03-17`

### Setup

```bash
git clone https://github.com/Filo-White/AGI_tree.git
cd AGI_tree

# Configure API key
cp .env.example backend/.env
# Edit backend/.env → OPENAI_API_KEY=sk-...

# Backend
conda create -n AGI_tree python=3.11 -y && conda activate AGI_tree
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --port 8000

# Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:3000**, upload a document, and start querying.

---

## Project Structure

```
AGI_tree/
├── backend/
│   ├── main.py                 # FastAPI endpoints + WebSocket
│   ├── tree_engine.py          # Tree construction, scoring, expansion, query routing
│   ├── llm_client.py           # OpenAI calls (classify, decompose, score, expand, answer)
│   ├── document_processor.py   # PDF/TXT text extraction
│   ├── models.py               # Pydantic schemas
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx             # Main app + AnalysisPanel
│   │   ├── types.ts            # TypeScript interfaces
│   │   └── components/
│   │       ├── TreeView.tsx    # Interactive tree visualization
│   │       ├── ChatPanel.tsx   # Query interface
│   │       └── NodeDetail.tsx  # Node inspection + manual expansion
│   └── vite.config.ts
├── experiments/
│   └── evaluation/             # Full evaluation framework (see below)
├── METODOLOGIA_ARCH.md         # Architectural methodology (IT)
├── DOCUMENTAZIONE.md           # Technical documentation (IT)
└── README.md
```

---

## Evaluation

The [`experiments/evaluation/`](experiments/evaluation/) directory contains the complete, reproducible evaluation framework:

- **Dataset**: 15 real-world documents across 7 categories, 122 annotated queries (15 unanswerable)
- **Methods**: DAT Full, 3 ablation variants, Flat RAG, Section-RAG, Long-Context
- **Metrics**: LLM-judge Likert scores (1–5), evidence routing (Hit@k, MRR), efficiency (tokens, latency, cost)
- **Judges**: Primary (`gpt-5.4-nano-2026-03-17`) + Secondary (`deepseek-v4-pro`) with inter-judge agreement (Spearman ρ = 1.0)
- **Statistical rigor**: Bootstrap 95% CIs (10,000 resamples), method anonymization
- **Outputs**: 8 paper-ready LaTeX tables, raw logs, judge verdicts, qualitative examples

See [`experiments/evaluation/REPRODUCIBILITY.md`](experiments/evaluation/REPRODUCIBILITY.md) for full hyperparameter specification.

---

## Architecture Highlights

- **Structure-aware decomposition** — Document type determines how top-level nodes are created (chapters for books, sections for papers, titles for regulations).
- **Self-scoring with LLM** — Each node evaluates its own relevance to the query, eliminating the need for a centralized router.
- **Lazy expansion** — Nodes are expanded into sub-sections only when needed, keeping the tree shallow and efficient.
- **Interpretable routing** — Every answer is traceable to a specific document section, enabling citation and verification.
- **Real-time feedback** — WebSocket streams processing progress (classification → decomposition → scoring → expansion → answer).

---

## Known Limitations

- Two-level architecture struggles with deeply nested regulatory documents (Title → Chapter → Article → Paragraph). Recursive multi-level expansion is planned.
- Expansion recall is ~76%: some queries that should trigger expansion do not.
- Evaluation uses LLM-as-judge; systematic biases are mitigated via anonymization and dual-judge validation but not eliminated.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{dat2026,
  title={Document Agent Tree: Agentic Document QA via Structure-Aware Hierarchical Routing},
  author={Filippo Bianchini},
  year={2026},
  url={https://github.com/Filo-White/AGI_tree}
}
```

---

## License

This project is for academic and research purposes.
