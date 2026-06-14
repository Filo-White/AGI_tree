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

- **122 annotated queries** across **15 documents** (3.5M characters total)
- 7 document categories: textbook (1), scientific papers (4), administrative (3), regulation (1), reports (4), manuals (2)
- 5 query types: factual, section-specific, cross-section, procedural, unanswerable (15)
- Languages: English (10), Italian (5)
- All queries include reference answers, expected answer points, and supporting section annotations
- 3 controlled author-created documents (†) with known ground-truth structure

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
├── prompts/                    # Exact prompt templates
│   ├── answer_generation.txt
│   ├── judge_rubric.txt
│   ├── node_scoring.txt
│   └── document_classification.txt
├── data/
│   ├── documents.json          # 15 documents metadata
│   ├── queries.jsonl           # 122 annotated queries
│   └── docs/                   # Source documents (PDFs)
├── outputs/
│   ├── runs/<RUN_ID>/          # Per-query JSON logs (7 methods × 122 queries)
│   └── judge_outputs/          # Raw judge verdicts (primary + secondary)
│       ├── primary/
│       └── secondary/
├── REPRODUCIBILITY.md          # Full reproducibility details
└── results/
    ├── *.csv                   # Aggregate metrics + bootstrap CIs
    ├── inter_judge_correlation.json
    ├── qualitative_examples.md
    ├── error_analysis.md
    └── latex_tables/           # 8 paper-ready tables
```

## Configuration

Set `OPENAI_API_KEY` in `backend/.env`. Key parameters in `config.yaml`:

- `models.answer_model`: LLM used for all answer generation (same across methods)
- `models.judge_model`: Primary LLM for evaluation judge
- `models.judge_model_secondary`: Secondary LLM for inter-judge agreement
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
- Full parameter specification in `REPRODUCIBILITY.md`
- Exact prompt templates in `prompts/`
- Raw judge outputs for both primary and secondary judges in `outputs/judge_outputs/`
- Bootstrap 95% confidence intervals (10,000 resamples) in `results/bootstrap_confidence_intervals.csv`
- Inter-judge agreement (Spearman ρ) in `results/inter_judge_correlation.json`
- Qualitative success/failure examples in `results/qualitative_examples.md`
