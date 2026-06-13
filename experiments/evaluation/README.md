# DAT Evaluation Framework

Evaluation pipeline for the **Document Agent Tree (DAT)** system. Compares DAT against baselines and ablation variants for structure-aware document question answering.

## Reproduction

```bash
# Generate the annotated query dataset
python experiments/evaluation/build_dataset.py

# Run all methods
python experiments/evaluation/run_all.py --config experiments/evaluation/config.yaml

# Evaluate answers with LLM judge
python experiments/evaluation/evaluate_answers.py --run-dir experiments/evaluation/outputs/runs/<RUN_ID>

# Compute aggregate metrics from raw logs
python experiments/evaluation/compute_metrics.py --run-dir experiments/evaluation/outputs/runs/<RUN_ID>

# Generate LaTeX tables
python experiments/evaluation/make_tables.py --from-csv
```

## Methods

| ID | Method | Description |
|----|--------|-------------|
| A  | **DAT Full** | Full Document Agent Tree pipeline |
| B  | **Flat RAG** | Standard chunk-based retrieval + generation |
| C  | **Section-RAG** | Structure-aware retrieval (uses DAT section detection, no agentic scoring) |
| D  | **Long-Context** | Full document in prompt |
| E  | **DAT-no-expansion** | DAT without lazy expansion |
| F  | **DAT-parent-only** | DAT with top-level nodes only |
| G  | **DAT-no-self-scoring** | DAT with embedding retrieval instead of LLM scoring |

## Dataset

- **74 annotated queries** across **9 documents**
- 4 document categories: textbook (1), scientific papers (4), administrative (3), report (1)
- 5 query types: factual, section-specific, cross-section, procedural, unanswerable
- All queries include reference answers, expected answer points, and supporting section annotations

## Evaluation Dimensions

1. **Answer Quality** — LLM judge scoring (correctness, faithfulness, completeness, relevance, abstention accuracy)
2. **Evidence Localization** — Hit@1, Hit@3, MRR, Recall, Precision
3. **Structural Processing** — Document type accuracy, node count error, section name and text coverage
4. **Lazy Expansion** — Precision, recall, F1 of expansion decisions; quality/cost trade-off
5. **Computational Efficiency** — LLM calls, token consumption, latency, estimated cost

## Directory Structure

```
experiments/evaluation/
├── config.yaml                 # Model and evaluation parameters
├── build_dataset.py            # Dataset construction
├── run_all.py                  # Run all methods
├── run_method.py               # Run single method
├── evaluate_answers.py         # LLM judge
├── compute_metrics.py          # Metrics aggregation
├── make_tables.py              # LaTeX table generation
├── requirements.txt
├── baselines/
│   ├── flat_rag.py
│   ├── section_rag.py
│   └── long_context.py
├── dat_wrappers/
│   ├── dat_full.py
│   ├── dat_no_expansion.py
│   ├── dat_parent_only.py
│   └── dat_no_self_scoring.py
├── metrics/
│   ├── answer_quality.py
│   ├── routing.py
│   ├── structure.py
│   ├── expansion.py
│   └── efficiency.py
├── data/
│   ├── documents.json
│   ├── queries.jsonl
│   └── docs/                   # Source documents (PDFs)
├── outputs/
│   └── runs/<RUN_ID>/          # Per-query JSON logs
└── results/
    ├── *.csv                   # Aggregate metrics
    ├── error_analysis.md
    └── latex_tables/           # Paper-ready tables
```

## Configuration

Set `OPENAI_API_KEY` in `backend/.env`. Key parameters in `config.yaml`:

- `models.answer_model`: LLM used for all answer generation (same across methods)
- `models.judge_model`: LLM for evaluation judge
- `models.embedding_model`: Embedding model for retrieval-based methods
- `dat.expansion_threshold`: Score threshold for lazy expansion (0.7)
- Temperature 0.0 for all scoring, answering, and judging

## Reproducibility

- All metrics are reproducible from saved per-query JSON logs
- All tables are reproducible from saved CSV results
- Random seed fixed at 42
- Temperature 0.0 throughout
- Run IDs are timestamped; previous runs are never overwritten
- Method names are anonymized before passing to the LLM judge
