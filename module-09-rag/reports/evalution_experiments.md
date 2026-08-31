# RAG Evaluation Report — module-09-rag

**Metric definitions** (from the eval harness):
- **Evidence recall** = `satisfied_evidence_requirements / total_evidence_requirements` — did retrieval surface the chunk(s) needed to answer?
- **Answer accuracy** = `satisfied_answer_requirements / total_answer_requirements` — did the generated answer contain the required facts?

## Summary Across Versions

| Version | Change introduced | Questions | Avg. Recall | Avg. Accuracy | Zero-Recall Qs |
|---|---|---|---|---|---|
| `0.1.0`  | Vector search only | 19 | 52.6% | 59.6% | 9 |
| `0.1.11` | + Query expansion | 20 | 55.0% | **50.0%** ⬇ | 9 |
| `0.1.21` | + Hybrid search (vector + keyword) | 15 | 66.7% | 66.7% | 5 |
| `0.1.23` | + Cross-encoder re-ranking (25 candidates(each query) → top 5 (user query)) | 22 | **75.0%** | **72.7%** | 5 |

*(Note: the eval dataset grew/changed slightly between runs — question counts range 15–22 — so treat the trend directionally rather than as a strict apples-to-apples delta.)*

**Headline result:** recall improved **+22.4 points** and accuracy improved **+13.1 points** from the first vector-only baseline to the current hybrid + re-ranked pipeline.

## What each change did

### `0.1.0` → `0.1.11`: Query expansion (regression observed)
Adding LLM-generated query paraphrases nudged recall up slightly (52.6% → 55.0%) but **accuracy dropped from 59.6% to 50.0%** — the worst accuracy of any version tested. Expanding to multiple query variants pulled in a wider, noisier candidate set; on several questions (e.g. *"According to the preamble of her grandfather's will in Clarissa Harlowe..."*) the model went from a partially-correct answer to a fully wrong one once more (and more varied) context was stuffed into the prompt. On another question (*"What name did Don Quixote choose for his horse?"*) the model went from answering correctly to refusing to answer ("context does not contain sufficient information") because the expanded queries diluted the candidate set with less relevant chunks.

**Takeaway:** more retrieval queries ≠ better answers if the extra candidates aren't more relevant — expansion alone increased noise without improving precision.

### `0.1.11` → `0.1.21`: Hybrid search (fixed exact-match misses)
Hybrid search (0.6 vector / 0.4 keyword, min-max normalized and merged) recovered recall lost to expansion and then some (55.0% → 66.7%), and brought accuracy back up to match (66.7%).

**Concrete example:** *"Who was the Bishop of D in 1815?"* — both vector-only (`0.1.0`) and query-expansion (`0.1.11`) failed to retrieve the required chunk (recall = 0, model responded "context does not mention 1815"). Pure semantic search was matching on the general topic but missing the specific named entity. Once keyword search was blended in, the exact chunk containing "Bishop of D——" and "1815" was retrieved, and the model answered correctly: *"In 1815, M. Charles-François-Bienvenu Myriel was the Bishop of D——."* This is the classic vector-search blind spot — proper nouns and exact phrases — that lexical matching closes.

### `0.1.21` → `0.1.23`: Cross-encoder re-ranking (fixed buried evidence)
Adding a re-ranker (retrieve 25 candidates(for each query) → cross-encoder re-score → keep top 5) pushed recall to 75.0% and accuracy to 72.7% — the best of all four versions, and the only version where `zero_recall_questions` dropped to its lowest point (5) alongside the highest `perfect_recall_questions` (16/22).

**Concrete example — the improvement referenced in the module notes:** *"What is the name of the fortress prison where Edmond Dantès is imprisoned?"* — this failed in **every prior version** (`0.1.0`, `0.1.11`, `0.1.21`), each time returning "the specific name... is not mentioned" even after hybrid search was added. The correct chunk was apparently present in the broader candidate pool but never made it into the final top-5 by raw similarity/keyword score alone — a case of **context dilution**, where the right evidence was outranked by more superficially similar-but-irrelevant chunks. After adding the cross-encoder re-ranker over a wider 25-candidate pool, the correct chunk was finally surfaced and the model answered correctly: *"the name of the fortress prison... is the **Château d'If**."* The same fix also resolved the Don Quixote horse-name question — `0.1.23` is the only version where the model's answer was actually grounded in the matched required chunk, rather than getting the right answer from the LLM's own parametric knowledge despite a recall miss.

**Takeaway:** once candidate pools got wider and more heterogeneous (vector + keyword), a cheap similarity score alone wasn't sufficient to rank the truly relevant chunk into the top-5 — re-ranking with a cross-encoder was the fix.

## Remaining gaps
Several questions still fail across all four versions (e.g. *"what habits did Mrs. Fortescue recount to Miss Howe"*, *"where was Javert born"*) — these look like multi-hop or narrative-detail questions where the required fact isn't confined to a single well-matched chunk, suggesting the next improvement to evaluate is either larger context windows per chunk or a multi-hop retrieval step, rather than further tuning of top-k or ranking.