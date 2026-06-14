# Error Analysis — DAT Evaluation

**Run ID:** 20260614_152300

---

## 1. Top Recurring Failure Modes

### 1.1 Administrative Document Over-Segmentation
DAT frequently over-segments administrative, legal, and regulatory documents. The university regulation (doc003), GDPR (doc008), and EU AI Act (doc010) were split into more nodes than the expected structural level. The EU AI Act is the most severe case: 13 nodes generated vs. 8 expected (node error = 5), with the deeply nested Title→Chapter→Section→Article hierarchy exceeding DAT's two-level architecture. The IPC-A-610F manual (doc013) also exhibited over-segmentation due to its multi-level section numbering.

**Impact:** Correctness drops by ~0.3 on administrative/regulatory documents compared to scientific papers.
**Frequency:** Observed in 4/15 documents (doc003, doc007, doc010, doc013).

### 1.2 False Negative Expansion
In approximately 24% of queries annotated as requiring expansion, DAT did not trigger lazy expansion. The most common reason: the best-scoring node had a score above 0.7 (expansion threshold) but the score reflected topical relevance rather than content granularity. The system correctly identified the right section but answered from the node-level summary, missing sub-section details.

**Impact:** Completeness drops by ~0.5 on affected queries.

### 1.3 Flat RAG Chunk Boundary Issues
Flat RAG frequently retrieved chunks that split content across boundaries. For procedural queries, relevant steps were often split across 2–3 chunks, with only one retrieved in top-k.

**Impact:** Completeness for Flat RAG on procedural queries averages 2.7/5.
**Frequency:** ~40% of procedural queries.

### 1.4 Unanswerable Question Hallucination
All methods occasionally generated substantive answers to unanswerable questions (15 unanswerable queries across the dataset). Flat RAG was worst (40% abstention accuracy), likely because retrieved chunks sometimes contained superficially related content triggering hallucination. DAT performed best (80%, 12/15) due to node-level scoring providing a stronger signal of content absence.

### 1.5 Document Type Misclassification
DAT misclassified 4/15 documents: the administrative doc003 was predicted as "report", the GDPR (doc008) as "report", the EU AI Act (doc010) as "administrative" instead of "regulation", and the Python tutorial (doc014) as "book" instead of "manual". The impact on answer quality was moderate for doc003/doc008 (over-segmentation) and minimal for doc014 (book and manual splitting strategies produce similar structures for linearly organized content).

---

## 2. Where DAT Outperforms Baselines

### Example 2.1 — Cross-Section Synthesis (doc002_q03)
**Query:** "How do model-based methods differ from model-free methods in RL?"
- **DAT Full:** Correctly identified and selected both "Model-Free Methods" and "Model-Based Methods" nodes, synthesized a comprehensive comparison (Correctness: 5, Completeness: 5).
- **Flat RAG:** Retrieved 3 chunks from "Model-Free Methods" and 1 from "Theoretical Foundations". Missed the "Model-Based Methods" section entirely (Correctness: 3, Completeness: 2).

**Reason:** DAT's structural awareness identifies that two separate sections are relevant. Flat RAG's chunk retrieval is biased toward sections with more keyword overlap.

### Example 2.2 — Procedural Query (doc003_q02)
**Query:** "Come si prenota una sala riunioni?"
- **DAT Full:** Correctly routed to "Spazi Studio e Biblioteche" node, expanded it, and found the booking procedure sub-section (Completeness: 4).
- **Flat RAG:** Retrieved chunks from three different sections, one partially overlapping. Answer was incomplete (Completeness: 2).

### Example 2.3 — Unanswerable Detection (doc004_q06)
**Query:** "Does the paper propose any method for document summarization?"
- **DAT Full:** All nodes scored below 0.3. System correctly abstained.
- **Flat RAG:** Retrieved chunks about "sequence-to-sequence" tasks, generated a hallucinated answer about summarization capabilities.

---

## 3. Where Baselines Outperform DAT

### Example 3.1 — Simple Local Factual (doc005_q06)
**Query:** "How many parameters does BERT-Large have?"
- **Flat RAG:** Retrieved the exact chunk containing "340 million parameters" (Correctness: 5).
- **DAT Full:** Correctly routed to the "BERT" node, but the node-level context (truncated) did not include the exact parameter count. Answer mentioned "hundreds of millions" without the precise number (Correctness: 4).

**Reason:** DAT's node context truncation can lose specific details that smaller, detail-rich chunks preserve.

### Example 3.2 — Long-Context on Short Documents
For documents under ~5000 tokens, long-context consistently matches or outperforms DAT because the overhead of tree construction, scoring, and routing does not provide added value when all content fits in a single prompt window.

---

## 4. Ablation Insights

### Self-Scoring vs. Embedding Retrieval
Removing LLM self-scoring (DAT-no-self-scoring) and replacing it with embedding similarity causes the largest quality drop (–0.47 correctness). This confirms that node-specific self-assessment is a key contributor to DAT's routing precision. Embedding similarity cannot capture the nuanced relevance judgment that LLM scoring provides.

### Expansion Value
DAT-no-expansion shows a measurable gap vs. DAT-full (–0.23 correctness), concentrated on queries requiring sub-section detail. The efficiency gain from skipping expansion (~900 fewer tokens/query, 0.9s less latency) does not compensate for the quality loss.

### Parent-Only
DAT-parent-only demonstrates that even without leaves, the agentic scoring approach outperforms flat retrieval. But the absence of fine-grained routing results in lower completeness (–0.43 vs. DAT-full) on section-specific queries.

---

## 5. Practical Interpretation

### Key Findings
1. **DAT provides consistent improvement** on answer quality (+0.89 correctness, +0.66 faithfulness over Flat RAG) and evidence localization (+0.34 Hit@1, +0.25 Hit@3 over Flat RAG). Bootstrap 95% CIs confirm non-overlapping intervals between DAT and all baselines on correctness.

2. **Structure-awareness matters:** Section-RAG outperforms Flat RAG on most metrics, confirming that document structure provides valuable signal. DAT's additional improvement over Section-RAG (~0.4 correctness) demonstrates the value of agentic self-scoring.

3. **Long-context is a strong baseline** for short documents and simple factual queries. It approaches DAT on faithfulness (4.05 vs 4.23) because it avoids information loss from routing. However, it lacks evidence localization capability and scales poorly with document length. Two independent LLM judges (gpt-5.4-nano and gpt-4.1-mini) produce the same method ranking (Spearman ρ = 1.0).

4. **Lazy expansion provides meaningful gains** when it triggers correctly (quality delta +0.38 on average), but current recall is ~76%. Improving the expansion trigger is a promising direction.

5. **Administrative and regulatory documents remain challenging** due to over-segmentation. The hierarchical Title → Article → Paragraph structure does not map cleanly to DAT's two-level node/leaf architecture. The EU AI Act (doc010) is the worst case with 5 excess nodes and complete over-segmentation. Manuals show mixed results: the IPC standard over-segments but the Python tutorial performs well.

6. **Cost trade-off is real but manageable:** DAT uses ~2.5× more tokens per query than Flat RAG, but with nano-class models the absolute cost remains low (~$0.003/query). The quality improvement justifies the overhead for high-stakes document QA.

### Limitations
- The dataset covers 122 queries across 15 documents; per-category breakdowns (especially for regulation, n=1) should be interpreted cautiously.
- The LLM judge may have systematic biases, although method anonymization and dual-judge validation mitigate this.
- Cross-document generalization beyond the tested categories has not been evaluated.
- Very large documents (>500k chars, e.g. IEA WEO, IPC-A-610F) may exhibit coverage gaps for cross-reference queries due to top-k constraints.
- See `qualitative_examples.md` for concrete success/failure cases.
