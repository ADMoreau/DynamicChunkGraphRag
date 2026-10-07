# Summary

This is a practical exploration. Can retrieval units learned from past questions (demand-driven
chunk breaking), plus a graph that links and traverses them, beat strong standard pipelines on
accuracy or cost? This page records what was tested and what held up, including the negative
results. `results/RUNLOG.md` maps every number here to its config, full log, output files and
cost.

## In one paragraph

On single-document and 2-hop benchmarks (SQuAD, SyllabusQA, HotpotQA), **neither learned chunk
breaking nor graph traversal beat a plain retriever with a small reranker on held-out questions**;
the tuning-set gains there reversed. **On MuSiQue (2–4 hops), where flat retrieval really fails to
find later hops, beam traversal over sentence units did hold up on held-out questions**: +0.077 in
how often every hop reached the reader at 512 tokens (see section 6). Two findings did hold up:

1. A small reranker over sentence-sized units is a strong, cheap default. On a GPU it is about
   2× cheaper to serve than Provence, the state-of-the-art context pruner, for 1.5–5 F1 points
   less on single-document questions.
2. Provence wins on single-document questions but discards multi-hop evidence. It prunes each
   passage against the question, so the second hop gets cut. A pipeline that decides per
   question whether to prune looks like the most promising practical direction.

   ## Setup

| | |
| --- | --- |
| Datasets | **SQuAD** (single paragraph). **HotpotQA** (2 hops; 19,269 pooled Wikipedia paragraphs). **SyllabusQA** (63 real course syllabi; questions about the same passages repeat, 66% share a gold sentence; retrieval scoped to the question's own syllabus). **MuSiQue** (2–4 hops) |
| Retrieval | BM25 + bge-small embeddings, fused by reciprocal rank; MiniLM cross-encoder reranker (23M parameters) |
| Readers | Qwen2.5-3B (CPU, llama.cpp) for SQuAD; Qwen2.5-7B (GPU, vLLM) for answer grading |
| Answer judge | Qwen2.5-14B (v2 prompt). It agrees with Claude's 100 hand labels 83 times and is only ever too strict, so absolute accuracy is understated equally for every method |
| Budgets | tokens of context the reader receives (32–1,024) |
| Measures | answer reached the reader; **every** answer part / hop reached the reader; reader F1 or judged answer accuracy; latency, GPU time and energy |
| Discipline | settings chosen on past-question tuning sets; claims made only on held-out questions, with paired bootstrap 95% intervals |

## Results by idea (held-out unless marked)

### 1. Learned chunk breaking

| Variant | Result |
| --- | --- |
| Learned sentence splits (SQuAD) | tied with plain sentences + reranker |
| Hierarchical cracking, paragraph → sentence → clause → phrase, per-level k (SQuAD, SyllabusQA) | indexed by their own text, tiny pieces can't be found (0.70 vs 0.88); **context-aware indexing** (piece + its sentence) fixes that. Overall a tie |
| **Evidence units** (SyllabusQA): sentence base; the exact evidence spans past answers used become one unit each (cut at the edges, rejoined across sentences, locked by demand) | **retrieval:** every answer span reached the reader +0.044 [+0.016, +0.073] more often at 64 tokens on repeated-demand questions. **Answers:** no significant change (+0.022 at 64, n.s.) |

The evidence-unit retrieval gain is real but doesn't turn into better answers: a fragment without
its surrounding context isn't enough for the reader.

### 2. Node merging and LLM adjudication

| Finding | Evidence |
| --- | --- |
| Embedding similarity alone is unsafe for "same thing" | best setting right only 57% of the time; merged Super Bowl XXXVIII with XXVIII, "three" with "four", "earlier" with "later" |
| A Qwen 14B judge is safe | on 237 reference pairs: 0 merges of genuinely different things, accuracy 0.81; its errors sit on the same / related line |
| Constraints are needed at scale | identical-name veto (172 SQuAD / 5,116 HotpotQA groups), cannot-link from related / different verdicts, entity-type guard |
| Cost | about $0.38 (SQuAD, 31K pairs) and about $2.30 (HotpotQA, 175K pairs) on a Modal L40S |

### 3. Graph traversal

| Method | Result |
| --- | --- |
| Personalized PageRank | no gain on SQuAD or HotpotQA: the seeds fill the budget before anything the walk finds |
| Two-hop expansion | no gain |
| **Beam search** (breadth b, depth d; moves to an adjacent piece or along a rare shared concept) | **tuning:** +0.098 answers (SyllabusQA, covered, 32 tokens) and +0.064 both hops (HotpotQA, 512). **Held-out:** SyllabusQA −0.035 / −0.042 at 128 / 256 (significant); HotpotQA −0.110 at 256 (significant), tie at 512 |
| Cost | walk +2 ms; beam about 2.2× the search time of reranking alone on CPU |

Why it didn't help: on these benchmarks retrieval already finds the evidence. In the HotpotQA pool,
flat retrieval had both gold paragraphs in its top 30 for 89% of questions, so the bottleneck is
choosing what fits the budget, not finding it. Traversal spends budget on neighbours that are
usually less relevant than the reranker's next pick.

### 4. Against the state of the art (Provence, top 5, titles kept, threshold 0.1)

| Setting (held-out) | Provence | Sentences / units + reranker | Note |
| --- | --- | --- | --- |
| SyllabusQA answers @64 / 128 / 256 | **0.482 / 0.567 / 0.611** | 0.442 / 0.521 / 0.577 | Provence +0.04–0.05, with fewer tokens |
| HotpotQA, both hops @256 / 512 | 0.220 / 0.230 | **0.475 / 0.635** | Provence keeps about 130 tokens and drops second-hop evidence |
| SQuAD reader F1 @128 (7B, GPU) | **0.855** | 0.840 | |
| Front-stage cost on an L4 GPU | 68 ms, 6.8 J per question | **19 ms, 0.27 J** | end to end: about 10% lower latency, 1.8–2.7× less GPU time and energy |

Fairness caveat: Provence ran at its paper's default threshold. Tuning it on the tuning set could
narrow the multi-hop gap.

### 5. Does scale create room for a graph?

Partly. Mixing the HotpotQA pool into random Wikipedia abstracts (corpus from BEIR/hotpotqa), the
share of bridge questions whose two gold paragraphs are both in the top 30 falls from **0.87
(19K paragraphs) to 0.68 (1M)**. The first hop stays easy (0.94 in the top 3). So at scale there
is headroom for linking. A held-out gain from our traversal would still have to be shown there.

### 6. MuSiQue (2–4 hops): the first held-out gain for graph traversal

MuSiQue is where flat retrieval really fails to find later hops: on tuning questions, sentences and
reranker got every hop to the reader for 0% of 4-hop questions at 512 tokens:

| Every hop reached the reader | @256 | @512 | Search (CPU) |
| --- | ---: | ---: | ---: |
| Sentences + reranker | 0.124 | 0.201 | 0.5 s |
| **Beam over sentence units (beam 5, depth 4)** | **0.175** | **0.279** | 3.1 s |
| Beam over cut-up units (mostly whole paragraphs here) | 0.150 | 0.226 | 8.4 s |
| Provence | 0.028 | 0.030 | 4.3 s |

The sentence-unit beam beats sentences + reranker by **+0.051 [+0.023, +0.079] @256** and **+0.077
[+0.051, +0.105] @512**, larger than on tuning. Most of the gain is on 2-hop questions (0.296 → 0.441);
4-hop goes from 0.000 to 0.025. This is evidence reaching the reader, not yet answer accuracy, and
it is relative to our small-model baselines (our passage recall@5 is 53.5, against 74.7 for HippoRAG 2).

A beam over **sentence-base cut-up units** (sentences everywhere plus learned finer cuts, k 1/2/3) performs
the same as the plain-sentence beam on held-out questions (−0.008 at 512, n.s.). It still beats
sentences + reranker by +0.069 at 512. MuSiQue questions rarely revisit the same passages, so the
learned cuts barely change the units (about 320 extra pieces among roughly 76,000 sentences): the gain
comes from the traversal, not from the cutting.

## What didn't replicate

Two tuning-set gains reversed on held-out questions (SyllabusQA beam, HotpotQA beam). Both came from
small tuning sets (132–140 questions) with several variants compared. **Treat any tuning-only
result in the run log as a hypothesis.**

## Practical recommendations

1. **Default:** sentence-sized units with a small cross-encoder reranker. It's cheap, robust, and
   competitive.
2. **When accuracy matters on single-document questions,** Provence-style pruning is worth about 4–5
   points at 3–6× the front-stage compute.
3. **For multi-hop questions, don't prune against the question.** Keep full units and a reranker.
   Deciding per question is the most promising thing to build next.
4. **Merging nodes:** use an LLM judge plus constraints, never raw embedding similarity.

## Caveats

- **Judge.** Absolute answer accuracy is understated (the judge marks 17% of correct answers wrong).
  Comparisons between methods are paired and unaffected.
- **Scale.** Corpora are 2K–21K paragraphs, plus a 1M-document retrieval-only check.
- **Reference labels.** Claude labelled the merge reference set (250 pairs) and the judge check
  (100 answers)

## References

- Contextual Retrieval: https://www.anthropic.com/engineering/contextual-retrieval
- PIC, Document Segmentation Matters for RAG: https://aclanthology.org/2025.findings-acl.422.pdf
- Provence: https://arxiv.org/html/2501.16214v1
- Dense X Retrieval: https://aclanthology.org/2024.emnlp-main.845
- AutoPrunedRetriever: https://arxiv.org/html/2602.04926
- HippoRAG 2: https://arxiv.org/abs/2502.14802
- MultiHop-RAG: https://arxiv.org/pdf/2401.15391
- Stochastic database cracking (design inspiration for incremental splitting):
  https://stratos.seas.harvard.edu/publications/stochastic-database-cracking-towards-robust-adaptive-indexing-main-memory
- IDC, Intent-Driven Dynamic Chunking: https://arxiv.org/abs/2602.14784
  (code: https://github.com/unseen1980/IDC)
- GAM-RAG, Gain-Adaptive Memory for Evolving Retrieval: https://arxiv.org/abs/2603.01783
- ReMindRAG: https://arxiv.org/abs/2510.13193
- CrackIVF, Cracking Vector Search Indexes: https://arxiv.org/abs/2503.01823
- Query association surrogates for web search (the idea behind B4):
  https://ideas.repec.org/a/bla/jamist/v55y2004i7p637-650.html
- KVzip (caution: a cache compressed for one question served later questions poorly):
  https://arxiv.org/abs/2505.23416