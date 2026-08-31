# Diagnosis & Fix: "What is the name of the fortress prison where Edmond Dantès is imprisoned?"

## The Failure

Under naive vector-only retrieval (`v0.1.0`), this query returned **recall = 0** and **accuracy = 0**. The system retrieved 5 chunks, none of which contained the required evidence (chunk `..._1265`), and the model responded:

> *"Based on the provided retrieved context, the specific name of the fortress prison where Edmond Dantès is imprisoned is not mentioned."*

The failure wasn't limited to the baseline — it persisted through `v0.1.11` (query expansion) and `v0.1.21` (hybrid search) with the exact same missing chunk and refusal. Two rounds of retrieval improvements failed to fix it, which was the signal that the problem wasn't *which search method* was being used — it was *how* candidates were being ranked.

## Failure Mode: Context Dilution (buried evidence)

This is a **ranking failure, not a coverage failure**. The correct chunk was very likely present somewhere in the broader candidate pool (both vector and keyword search cover named entities like "Château d'If" reasonably well), but it never scored high enough by raw cosine similarity or keyword overlap to land in the final top-5. It was outranked by chunks that were topically similar — other passages about Dantès, imprisonment, or the ship/voyage — but didn't actually answer the question. More candidates and a second retrieval channel (hybrid) didn't help, because the bottleneck was precision at the top of the list, not recall within the pool.

## The Fix: Cross-Encoder Re-Ranking

In `v0.1.23`, a re-ranking stage was added: retrieve a wider candidate pool (**top_k(25*4) = 100**) using hybrid search, then re-score all 100 with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`) and keep only the **top 5** by that finer-grained relevance score. Unlike bi-encoder similarity, a cross-encoder jointly attends over the (query, chunk) pair, so it can distinguish "topically related" from "actually answers the question" — exactly the distinction this query needed.

## Before / After

| Version | Retrieval | Recall | Accuracy | Generated Answer |
|---|---|---|---|---|
| `v0.1.0` | Vector only | 0.0 | 0.0 | "...is not mentioned." |
| `v0.1.11` | + Query expansion | 0.0 | 0.0 | "...is not mentioned." |
| `v0.1.21` | + Hybrid search | 0.0 | 0.0 | "...is not mentioned." |
| `v0.1.23` | + **Cross-encoder re-rank** (100→5) | **1.0** | **1.0** | *"the name of the fortress prison... is the **Château d'If**."* |

Re-ranking was the only change of the three that flipped this query from a hard failure to a fully grounded, correct answer — confirming the root cause was ranking precision, not retrieval coverage.