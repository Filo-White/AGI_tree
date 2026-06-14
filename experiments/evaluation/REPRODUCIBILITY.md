# Reproducibility Details

Complete specification of all parameters, models, and configurations used in the DAT evaluation.

## Models

| Component | Model | Temperature | Notes |
|-----------|-------|-------------|-------|
| Answer generation | `gpt-5.4-nano-2026-03-17` | 0.0 | All methods use the same answer model |
| Primary judge | `gpt-5.4-nano-2026-03-17` | 0.0 | Method names anonymized (Method_A–G) |
| Secondary judge | `gpt-4.1-mini-2025-04-14` | 0.0 | Same rubric, independent evaluation |
| Embedding | `text-embedding-3-small` | — | Used for Flat RAG chunking and DAT-no-self-scoring |

## RAG Baseline Parameters

| Parameter | Value |
|-----------|-------|
| Chunk size | 900 tokens |
| Chunk overlap | 120 tokens |
| Top-k retrieval | 4 chunks |

## DAT Parameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Top-k respondents | *k* | 4 | Maximum nodes scored per query |
| Score threshold | α | 0.3 | Minimum score for a node to be considered relevant |
| Expansion threshold | τ | 0.7 | Auto-expand when best score < τ AND node passes relevance check |
| Max respondent nodes | — | 4 | Hard cap on nodes returned per query |
| Lazy expansion | — | enabled | Nodes start as top-level only; leaves created on demand |
| Query decomposition | — | enabled | Complex queries split into sub-queries |

### Expansion Criteria

Auto-expansion triggers when **all** of the following hold:
1. `max(node_scores) < τ` (no node is confidently relevant)
2. The best-scoring node passes a secondary relevance check (prevents expanding irrelevant nodes)
3. The node has not already been expanded

When triggered, the best-scoring node is expanded into sub-sections (leaves), which are then scored independently. The final answer is generated from the highest-scoring leaf.

## Cost Model

Costs are estimated using OpenAI API pricing as of March 2026:

| Model | Input | Output |
|-------|-------|--------|
| `gpt-5.4-nano-2026-03-17` | $0.15 / 1M tokens | $0.60 / 1M tokens |
| `text-embedding-3-small` | $0.02 / 1M tokens | — |

Per-query costs include all LLM calls (classification, scoring, expansion, answer generation) and embedding calls.

## LLM Judge Configuration

- **Rubric**: 4-dimension Likert scale (1–5): Correctness, Faithfulness, Completeness, Relevance
- **Anonymization**: Method names replaced with random labels (Method_A through Method_G) to prevent judge bias
- **Prompt**: See `prompts/judge_rubric.txt` for the exact prompt template
- **Abstention evaluation**: For unanswerable queries (*n* = 15), we check whether the method correctly abstains (responds that the document does not contain the answer)
- **Inter-judge agreement**: Spearman ρ = 1.000 between primary and secondary judge rankings

## Random Seed

All stochastic components use seed `42`:
- Query shuffling
- Bootstrap resampling (10,000 iterations)
- Method presentation order for judge

## Dataset

- **15 documents** across 7 categories (textbook, scientific paper, administrative, report, regulation, manual, standard)
- **122 queries** (107 answerable, 15 unanswerable)
- 5 query types: factual, section-specific, cross-section, procedural, unanswerable
- All queries include reference answers, expected answer points, and supporting section annotations
- Full query set available in `data/queries.jsonl`

### Document Versions

| Doc ID | Document | Source | Version/Date |
|--------|----------|--------|-------------|
| doc001 | Introduzione all'IA | Author-created | v1.0 |
| doc002 | RL in Robotics Survey | Author-created | v1.0 |
| doc003 | Regolamento Campus | Author-created | v1.0 |
| doc004 | Attention Is All You Need | arXiv:1706.03762 | v7, 2023-08-02 |
| doc005 | BERT | arXiv:1810.04805 | v2, 2019-05-24 |
| doc006 | GPT-3 | arXiv:2005.14165 | v4, 2020-07-22 |
| doc007 | Regolamento Didattico | Univ. regulation | 2023 edition |
| doc008 | GDPR Excerpts | EUR-Lex 2016/679 | Consolidated 2016 |
| doc009 | World Energy Outlook 2023 | IEA | 2023 edition |
| doc010 | EU AI Act | EUR-Lex OJ L 2024/1689 | Published 2024-07-12 |
| doc011 | NIST AI RMF | NIST AI 100-1 | v1.0, January 2023 |
| doc012 | WHO Physical Activity Plan | WHO | 2018 |
| doc013 | IPC-A-610F | IPC | Revision F |
| doc014 | Il Tutorial di Python | python.org | Python 3.x (IT translation) |
| doc015 | IPCC AR6 WG1 SPM | IPCC | 2021 |

## Bootstrap Confidence Intervals

95% CIs computed via percentile bootstrap with 10,000 resamples over 122 queries. Results in `results/bootstrap_confidence_intervals.csv`.

## Prompt Templates

All prompt templates are available in `prompts/`:
- `answer_generation.txt` — Answer generation prompt
- `judge_rubric.txt` — LLM judge evaluation rubric
- `node_scoring.txt` — DAT node relevance scoring
- `document_classification.txt` — Document type classification
