# Qualitative Examples

Selected success and failure cases from the evaluation, illustrating DAT's strengths and weaknesses across document types.

## Success Cases

### Example 1: Cross-section query on scientific paper (doc006, GPT-3)

**Query**: "What are the risks of misuse discussed in the GPT-3 paper, and how do they relate to the model's few-shot capabilities?"

**DAT Full** (Correctness: 5, Faithfulness: 5):
> The paper identifies several risks of misuse directly linked to GPT-3's few-shot capabilities: generating fake news and social engineering content becomes easier as the model requires only a few examples to produce convincing text; spam and phishing can be automated at scale; and academic dishonesty is enabled by the model's ability to generate coherent essays. The paper notes that these risks scale with model capability — the same few-shot learning that makes GPT-3 useful for legitimate tasks also lowers the barrier for malicious use.

**Flat RAG** (Correctness: 3, Faithfulness: 3):
> GPT-3 can generate fake news, spam, and social engineering content. The model has few-shot capabilities that allow it to...

*Analysis*: DAT correctly routed to both the "Broader Impacts" and "Approach" sections, synthesizing the connection between capabilities and risks. Flat RAG retrieved chunks from only one section, missing the cross-reference.

---

### Example 2: Procedural query on regulation (doc010, EU AI Act)

**Query**: "What steps must a provider follow to place a high-risk AI system on the EU market?"

**DAT Full** (Correctness: 5, Completeness: 5):
> A provider must: (1) implement a quality management system covering design, development, and testing; (2) prepare technical documentation before market placement; (3) undergo a conformity assessment — either self-assessment for Annex VI systems or third-party assessment via a notified body for biometric systems; (4) register the system in the EU database; (5) affix the CE marking; (6) draw up an EU declaration of conformity; and (7) establish a post-market monitoring system proportionate to the AI system's risk.

**Section-RAG** (Correctness: 4, Completeness: 3):
> Providers must prepare technical documentation and undergo conformity assessment. The system must be registered in the EU database and carry the CE marking. Post-market monitoring is required.

*Analysis*: DAT's lazy expansion was triggered on the "High-Risk AI Systems" node (initial score 0.65 < τ=0.7), expanding to sub-sections covering obligations, conformity assessment, and post-market monitoring. This yielded a more complete procedural answer.

---

### Example 3: Factual query on manual (doc013, IPC-A-610F)

**Query**: "Quali sono le tre classi di accettabilità definite dallo standard IPC-A-610?"

**DAT Full** (Correctness: 5, Faithfulness: 5):
> Le tre classi di accettabilità definite dall'IPC-A-610 sono: Classe 1 (Prodotti Elettronici Generali), destinata a prodotti con requisiti estetici e funzionali base; Classe 2 (Prodotti Elettronici per Servizio Dedicato), per prodotti con esigenze di affidabilità e durata più elevate; Classe 3 (Prodotti Elettronici ad Alta Affidabilità), per prodotti dove il malfunzionamento è inaccettabile (dispositivi medici, aerospaziale, militare). I criteri di accettabilità diventano progressivamente più stringenti dalla Classe 1 alla Classe 3.

*Analysis*: DAT correctly classified the IPC manual and routed directly to the "Programma di Accettabilità" node, which contained the class definitions. No expansion was needed (score 0.92 > τ).

---

## Failure Cases

### Failure 1: Over-segmentation on regulation (doc010, EU AI Act)

**Document type**: Regulation (EU AI Act, 144 pages)
**Expected nodes**: 8 (corresponding to Titles I–VIII)
**Generated nodes**: 13
**Type classification**: Predicted "administrative" instead of "regulation"

DAT over-segmented the EU AI Act because its hierarchical Title → Chapter → Section → Article structure exceeds DAT's two-level node/leaf model. The classifier confused the numbered-article format with administrative documents. This led to fragmented nodes that split related provisions across multiple tree entries, reducing answer completeness for cross-reference queries.

---

### Failure 2: Abstention failure on unanswerable query (doc012, WHO)

**Query**: "Does the plan address cognitive enhancement through exercise in elderly populations?"

**Expected**: Abstention (the document does not specifically address cognitive enhancement)

**DAT Full response**: "The plan recommends regular physical activity for older adults and describes health benefits including improved functional capacity and reduced fall risk. Programs should be tailored to older adults' abilities and preferences."

*Analysis*: DAT found a moderately relevant node ("Active People") and generated a plausible but non-responsive answer about general benefits for older adults, failing to abstain. The node's relevance score (0.58) was below τ but the relevance check passed, triggering expansion that surfaced age-related content — which was tangentially related but did not address cognitive enhancement specifically.

---

### Failure 3: Python tutorial misclassified as book (doc014)

**Document type**: Manual (Python Tutorial IT)
**Expected classification**: manual
**Predicted classification**: book

The Italian Python tutorial's chapter-based structure (Cap. 1, Cap. 2, ...) with progressive topic ordering resembles a textbook more than a reference manual. DAT classified it as "book" rather than "manual", leading to slightly different section detection heuristics. Impact on answer quality was minimal (Correctness: 4.2 average for doc014 queries), as both book and manual splitting strategies produce similar node structures for linearly organized content.

---

### Failure 4: Long document coverage gap (doc009, IEA WEO)

**Query**: "What are the projected costs of different energy sources by 2050?"

**DAT Full** (Completeness: 2):
> The document discusses energy investment trends and policy implications but the specific cost projections by 2050 are distributed across multiple sections including Electricity, Fuels, and Energy Investment chapters.

*Analysis*: The IEA WEO document (916k chars, 355 pages) exceeded the practical context window even with DAT's selective retrieval. The relevant information was spread across 4+ nodes, and DAT's top-k=4 constraint meant some relevant content was excluded. This represents a scalability limitation for very large documents with highly distributed information.

---

## Summary of Patterns

| Pattern | Affected Doc Types | Frequency | Impact |
|---------|--------------------|-----------|--------|
| Over-segmentation | Regulation, Administrative | 4/15 docs | Moderate: fragments related content |
| Type misclassification | Regulation, Manual | 3/15 docs | Low: minimal impact on answer quality |
| Abstention failure | All types | ~20% of unanswerable | Moderate: false positive answers |
| Coverage gap on long docs | Reports > 500k chars | 2/15 docs | High for cross-reference queries |
