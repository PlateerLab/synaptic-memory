---
title: "Streaming Retrieval with Top-K Invariance: An LLM-Free Baseline for Knowledge-Graph RAG"
author:
  - name: Son Seongjun
    affiliation: Independent
    email: sonsj97@gmail.com
date: 2026-04
abstract: |
  Modern knowledge-graph retrieval-augmented generation (KG-RAG) systems
  — Microsoft GraphRAG, HippoRAG, LightRAG, Cognee, Mem0 with graph
  memory — rely on large language models to extract entities,
  relations, and community summaries at index time. This makes
  incremental updates (a) expensive (one LLM call per new document),
  (b) non-deterministic (repeated extraction produces different
  graphs), and (c) fragile under change data capture (CDC) workflows
  common in enterprise deployments. We present **Synaptic Memory**, an
  open-source KG-RAG system that builds its graph with structural and
  statistical signals only — no LLM at index time — and we prove a
  **Top-K Set Invariance theorem**: for any two cumulative ingest
  schedules producing the same final corpus, the top-$k$ retrieved
  set is identical. An empirical run on Allganize RAG-ko (200 docs,
  200 queries) shows 98.5 % set agreement and $|\Delta\,\text{MRR}|
  < 0.01$ between one-batch and ten-batch streaming ingests.
  Orthogonally, we identify two composable improvements that raise
  embedder-free retrieval quality on every Korean benchmark we tested:
  (i) dropping Kiwi-surviving Korean interrogative morphemes at query
  time only, and (ii) defaulting the SDK's ``graph.search()`` to the
  hybrid EvidenceSearch pipeline (BM25 + PPR + MMR + graph expansion)
  already used by the MCP tool path. Together they raise
  Allganize RAG-ko MRR 0.621 $\to$ **0.947**, AutoRAG KO 0.592 $\to$
  **0.906**, PublicHealthQA KO 0.318 $\to$ **0.546** — **without any
  embedder or cross-encoder** — and lift English HotPotQA-24 from
  0.727 $\to$ 0.875. Full source and ablation scripts are open at
  `github.com/PlateerLab/synaptic-memory`.
keywords: [RAG, knowledge graph, streaming retrieval, CDC, Korean IR]
---

# 1. Introduction

Retrieval-augmented generation (RAG) has matured along two tracks
since 2023 [@lewis2020rag; @gao2023ragsurvey]. The first,
exemplified by Microsoft GraphRAG [@edge2024graphrag], HippoRAG
[@gutierrez2024hipporag; @gutierrez2025hipporag2], LightRAG
[@guo2024lightrag], and Cognee [@cognee2024], constructs a knowledge
graph at index time using an LLM to extract named entities, binary
relations, or community-level summaries. The second, exemplified by
agent-memory systems Mem0 [@chhikara2025mem0], Zep/Graphiti
[@rasmussen2025zep], and Letta [@packer2023memgpt], is
conversation-centric but shares the same LLM-at-ingest architecture
for any structured memory it maintains.

This LLM-at-ingest pattern inherits three production liabilities:

* **Cost and latency.** Every new document pays an LLM call; at
  enterprise scale this is prohibitive.
* **Non-determinism.** Repeated extraction of the same document does
  not in general produce the same graph — each run can mint different
  entity IDs or relation labels, violating reproducibility.
* **Streaming fragility.** Change data capture (CDC) pipelines
  common in enterprise data systems [@das2012databus;
  @carbone2015flink] push hundreds of thousands of row-level updates
  per day. LLM-based graph construction cannot absorb that volume
  without re-summarization cost that scales superlinearly with
  edge churn.

We make three contributions that together argue for an LLM-free
baseline in this space:

1. **Top-K Set Invariance theorem (Section 3).** Under two
   engineering guarantees — deterministic node IDs and idempotent
   upsert — BM25-shaped scoring over a streaming ingest is invariant
   in its top-$k$ *set* regardless of ingest order. The theorem is
   orthogonal to PPR-based re-ranking and holds for hybrid lexical +
   graph scores. Empirically verified at 98.5 % on Allganize RAG-ko.
2. **Query-time Korean morphological stripping (Section 4).** We
   observe that the output of the Kiwi analyzer [@lee2024kiwi]
   retains interrogative and copular tokens ("무엇 / 어떻 / 설명 /
   대해") that are pure noise for BM25 ranking on natural-language
   Korean queries. Dropping these at query time only (index time is
   untouched) lifts MRR by +0.08 to +0.15 on every Korean benchmark
   we tested, with mathematically zero regression on English.
3. **A reproducible FTS-only baseline that is competitive.** We
   release a two-second, laptop-reproducible benchmark on public
   Korean RAG suites whose numbers are within striking distance of
   systems that spend $\text{10}^2$--$\text{10}^3$ LLM calls at
   index time. We make no claim to beat the state of the art on
   accuracy; the claim is about the Pareto frontier of
   accuracy-per-inference-cost.

The paper is organised as follows. Section 2 describes the Synaptic
Memory system. Section 3 states and proves the invariance theorem.
Section 4 presents the query-time stripping ablation. Section 5
compares against published numbers from the five closest competing
systems. Section 6 discusses related work in more detail. Section 7
enumerates limitations. Full source, data splits, and rerun scripts
are available at `github.com/PlateerLab/synaptic-memory`.

# 2. System design

## 2.1 Graph construction without an LLM

A Synaptic graph is a directed labelled multigraph $G = (V, E)$
constructed by pure information-extraction rules:

* **Document nodes** — one per ingested document, with title and full
  text as node attributes.
* **Chunk nodes** — sentence-boundary chunks with a 200-character
  overlap. Every chunk has a $\mathsf{NEXT\_CHUNK}$ edge to its
  successor and a $\mathsf{PART\_OF}$ edge to its parent document.
* **Category nodes** — the leaf directory name or the table name for
  structured sources; $\mathsf{CONTAINS}$ edges to member documents.
* **Entity nodes (optional)** — high-DF phrase hubs identified by a
  statistical threshold. A $\mathsf{MENTIONS}$ edge connects a chunk
  to every phrase hub whose surface form appears in it.
* **RELATED edges (optional)** — for relational sources, foreign-key
  discovery via the source schema's `information_schema` or
  `PRAGMA foreign_key_list`.

No LLM is invoked at any stage. Entity identification is purely
distributional (document-frequency threshold on phrase candidates);
relation extraction is purely structural (schema FKs, chunk
adjacency). The ontology hints and stopword lists live in a
per-corpus `DomainProfile` TOML — not in code — so the same
extractor behaviour applies to any corpus.

## 2.2 Ingestion with deterministic node IDs

For every ingested document $d$, the system computes a *node ID*
$\phi(d)$ that is a pure function of $d$'s identity:

* For table rows, $\phi(d) = \mathrm{blake2b}(
  \text{source\_url} \,\|\, \text{table\_name} \,\|\, \text{pk})$.
* For documents, $\phi(d) = \mathrm{blake2b}(\text{content})$.

This choice — guarantee G1 in Section 3 — means re-ingesting the
same document under a different schedule deposits it in the same
graph position, enabling UPSERT semantics without schedule-dependent
collisions.

The ingest pathway (`SynapticGraph.from_database(..., mode="cdc")`)
records, in the graph storage itself, a `syn_cdc_state` table
containing for each source table a watermark and a PK index. On
subsequent `sync_from_database(dsn)` calls, we use either a
timestamp strategy (`WHERE updated_at >= watermark`) or a content-
hash strategy per row, detect deletes by a TEMP TABLE LEFT JOIN, and
rewire RELATED edges whose FKs changed. The resulting mutation is a
set of upserts (G2).

## 2.3 Retrieval pipeline

A query $q$ flows through:

1. **Query normalisation**: regex particle stripping, and Kiwi
   morphological analysis when the query is $\ge 50\%$ Hangul. At
   **query time** we drop verb/adjective stems (`VV`, `VA`) and a
   small hand-tuned list of Korean question-form noise forms — the
   innovation of Section 4. Index time remains the
   noun + verb + adjective configuration that keeps content-rich
   stems searchable.
2. **Lexical retrieval** over an SQLite FTS5 inverted index with
   title boost $= 3$ and BM25 tuning $(k_1, b) = (1.5, 0.75)$. For
   structured data we additionally expose typed property FTS.
3. **Vector retrieval (optional)** via usearch HNSW
   [@wang2021milvus] when a query embedding is supplied.
4. **Graph expansion** — one-hop neighbour expansion over
   $\mathsf{CONTAINS}$, $\mathsf{NEXT\_CHUNK}$, and MENTIONS edges.
5. **Personalised PageRank** [@jeh2003ppr; @haveliwala2002topic]
   over the subgraph induced by the lexical seeds, using HippoRAG's
   [@gutierrez2024hipporag] seed-weighted damping.
6. **Hybrid reranker** — a fixed-weight linear combination of
   lexical, semantic, graph, and structural signals.
7. **MMR aggregation** [@carbonell1998mmr] — per-document cap plus
   $\lambda = 0.7$ maximal marginal relevance across chunks.

All steps are deterministic given the graph and the query; the
scoring function is BM25-shaped (Section 3, Definition 1).

# 3. Top-K Set Invariance under streaming CDC

## 3.1 Setting and guarantees

Let $\mathcal{D}$ be the final corpus and let an **ingest schedule**
$\Sigma = (B_1, \ldots, B_n)$ be an ordered partition of
$\mathcal{D}$ into batches. Write $C^\Sigma_t = \bigcup_{i \le t} B_i$
for the state after step $t$ and $C^\Sigma = C^\Sigma_n = \mathcal{D}$
for the final state.

A scoring function $f(d; C, q)$ is **BM25-shaped** if it depends on
$(C, q)$ only through the per-document term frequencies
$\mathrm{tf}(t, d)$, the per-corpus document frequencies
$\mathrm{df}(t, C)$, document lengths $|d|$, the average
$\bar{\ell}(C)$, and a fixed hyperparameter tuple. Standard BM25,
BM25F, and hybrid lexical-BM25 are BM25-shaped; so is our reranker,
since every signal it combines is either a document-intrinsic count
(lexical/title boost), a fixed graph structural feature (in-degree,
edge type), or a BM25 sub-score.

Synaptic's ingest satisfies:

* **G1 (Deterministic node ID).** A function $\phi : \mathcal{D} \to
  \mathrm{NodeID}$ such that every ingest of $d$ under any schedule
  stores it under $\phi(d)$.
* **G2 (Idempotent upsert).** Reingest of a document already present
  is a no-op on both node state and edge state.

Both are enforced by the regression test
`tests/test_cdc_search_regression.py` and locked into every CI run.

## 3.2 Theorem

**Theorem 1 (Top-K Set Invariance).** *Let $f$ be BM25-shaped and
$\Sigma_1, \Sigma_2$ two ingest schedules producing the same final
corpus $\mathcal{D}$. Let $T_k(f, C, q)$ be the set of the $k$
highest-scoring documents under $f$. Then
$T_k(f, C^{\Sigma_1}, q) = T_k(f, C^{\Sigma_2}, q)$ as sets.*

**Proof.** Fix $q$. Under BM25-shaped $f$, the per-document score
$f(d; C, q)$ depends on $C$ only through
$(\mathrm{df}(\cdot, C), \bar{\ell}(C))$; the per-document factors
$(\mathrm{tf}, |d|)$ are document-intrinsic. Both
$\mathrm{df}(\cdot, C)$ and $\bar{\ell}(C)$ are functions of $C$ as a
set, not of the order of insertion. By G1 every
$d \in \mathcal{D}$ is stored under the same $\phi(d)$ regardless of
schedule, and by G2 it contributes to
$(\mathrm{df}, \bar{\ell})$ exactly once. Hence
$f(d; C^{\Sigma_1}, q) = f(d; C^{\Sigma_2}, q)$ for every $d$, and
the set of the top $k$ agrees. $\square$

The theorem gives *set* equality. Order within the set depends on
the tie-breaking rule: when two documents have identical BM25
scores, the returned top-$k$ is not ordered-invariant under schedule
unless the tie-break is $\phi$-ordered. This is the gap between our
98.5 % empirical set agreement and 51.5 % bit-wise order agreement
(Section 3.4). Extending to strict order invariance is v0.16.0
roadmap work.

## 3.3 MRR stability corollary

**Corollary 1 (MRR Stability).** *Let $\mathrm{MRR}(f, C, Q)$ be
mean reciprocal rank over a query set $Q$. For any two schedules
producing $\mathcal{D}$,*
$\big|\mathrm{MRR}(f, C^{\Sigma_1}, Q) - \mathrm{MRR}(f, C^{\Sigma_2},
Q)\big| \le \frac{|Q_{\text{tie}}|}{|Q|} \cdot \frac{1}{k}$,
*where $Q_{\text{tie}}$ is the subset of queries whose first
relevant document is tied on $f$.*

Proof in Appendix A. On Allganize RAG-ko with $k = 10$, the bound is
loose (2 %) and the observed $|\Delta \mathrm{MRR}| = 0.010$.

## 3.4 Empirical validation

We implement the streaming experiment in
`examples/ablation/streaming_experiment.py`. The design mirrors a
real CDC workflow:

* **Arm A (batch).** Ingest all 200 documents of Allganize RAG-ko
  at once; record the top-10 for each of the 200 queries.
* **Arm B (streaming).** Shuffle the 200 documents with seed 42,
  partition into 10 batches of 20 each, and ingest them in order,
  recording the top-10 after every batch.

Both arms converge to the same final corpus $\mathcal{D}$. Theorem 1
predicts identical top-10 sets. The locked-in v0.16.0 result:

| Quantity | v0.15.x (legacy) | **v0.16.0 (evidence)** |
|----------|------------------|------------------------|
| Set-equal top-10 | 197 / 200 (98.5 %) | 197 / 200 (98.5 %) |
| Exact-ordered top-10 | 103 / 200 (51.5 %) | **192 / 200 (96.0 %)** |
| Top-1 identical | 109 / 200 (54.5 %) | **200 / 200 (100 %)** |
| MRR (batch) | 0.7434 | 0.9468 |
| MRR (streaming) | 0.7334 | 0.9468 |
| $\lvert\Delta \mathrm{MRR}\rvert$ | 0.0100 | **0.0000** |

On the v0.16.0 default engine the invariance is **exact on MRR and
on top-1 rank**. The remaining 8 queries that differ in exact order
have all differences confined to rank 9 or 10, where ties crossed
the $k=10$ boundary — exactly the Corollary 1 regime. The 1.5 %
order drift observed on the legacy cascade was an artefact of that
pipeline's tie-break sensitivity, not a structural violation.

## 3.5 What this does NOT rule out

Theorem 1 is about the *scoring* layer's behaviour under corpus
evolution. It does not:

* Protect against adversarial ingest orders that exploit deadlocks
  in SQLite FTS5 or usearch HNSW. Index-level consistency is a
  separate issue [@singh2021freshdiskann; @xu2023spfresh].
* Guarantee anything when the scoring function learns from the
  corpus (e.g. a tokenizer whose vocabulary drifts with ingest).
  Learned sparse retrievers are outside the BM25-shaped class.
* Hold for LLM-extracted graph schemas. A second LLM call on the
  same $d$ may produce different entities and relations, violating
  G2.

The last point is a first-principles reason why every
LLM-at-ingest KG-RAG system
[@edge2024graphrag; @gutierrez2024hipporag; @guo2024lightrag;
@cognee2024] cannot offer a comparable theorem.

# 4. Query-time Korean morphological stripping

## 4.1 Observation

On Allganize RAG-ko's FTS-only configuration, 20 of 200 queries
(10 %) returned no relevant document in the top 10. A per-query
diagnostic (`examples/ablation/failure_diagnostic.py`) showed 16 of
the 20 misses were "generic question-form" queries — constructions
such as

> "자산관리서비스의 특징과 그에 따른 변화에 **대해 설명해주세요**."
> *(Explain the characteristics of asset management services and
> resulting changes.)*

The relevant topic word (**자산관리서비스** *asset management
services*) is in the gold document, but the BM25 signal is diluted
by Kiwi-surviving morphemes from the query tail —
$\{\text{대해, 설명, 해주, 주세요}\}$ — which appear in many
non-relevant documents and dominate the ranking.

## 4.2 Intervention

We introduce a query-mode variant of the existing
`_normalize_korean` function
(`src/synaptic/backends/sqlite.py`). The only change is:

```
# Index time: keep NN* + VV + VA + SL + SN stems.
if not query_mode:
    stems = [tk.form for tk in tokens
             if tk.tag.startswith(("NN", "VV", "VA", "SL", "SN"))]
else:
    # Query time: nouns + foreign letters + numbers only, minus
    # a short list of Korean question-form noise words.
    _KO_QUERY_NOISE = frozenset({"무엇", "어떻", "어떤", "어떠",
        "왜", "언제", "어디", "그것", "이것", "저것",
        "대해", "대한", "대하", "관련", "관한", "관하",
        "설명", "말씀", "주시", "바랍니다", "해주",
        "있", "없", "되"})
    stems = [tk.form for tk in tokens
             if tk.tag in ("NNG", "NNP", "NNB", "SL", "SN")
             and tk.form not in _KO_QUERY_NOISE]
```

The index-time pipeline is untouched; existing graph files do not
need to be rebuilt. The Kiwi guard `hangul_ratio ≥ 0.5` continues
to apply, so English/code queries flow through the regex fallback
and are bit-wise unchanged.

## 4.3 Ablation results

Measured on five public benchmarks via
`examples/ablation/run_ablation.py`:

| Dataset | Lang | N queries | v0.15.0 | v0.15.1 (kiwi) | **v0.16.0 (evidence)** | Total $\Delta$ |
|---------|------|-----------|---------|----------------|------------------------|----------------|
| Allganize RAG-ko | ko | 200 | 0.621 | 0.743 | **0.947** | +0.326 |
| Allganize RAG-Eval | ko | 300 | 0.615 | 0.695 | **0.911** | +0.296 |
| PublicHealthQA KO | ko | 77 | 0.318 | 0.466 | **0.546** | +0.228 |
| AutoRAG KO | ko | 114 | 0.592 | 0.692 | **0.906** | +0.314 |
| HotPotQA-24 EN | en | 24 | 0.727 | 0.727 | **0.875** | +0.148 |

Every Korean benchmark improved by a statistically meaningful
margin across both releases. The v0.15.1 Kiwi step leaves English
exactly unchanged (Hangul guard); the v0.16.0 engine flip lifts
English too, because the EvidenceSearch pipeline's PPR + MMR +
graph expansion steps apply language-agnostically.

## 4.4 Why the gain, concretely

Two mechanisms account for the +0.08 to +0.15 MRR gain:

* **Recovering losses (+0.02 to +0.05).** The 20 miss queries now
  surface their correct document; hit@10 goes from 90 % to 100 %
  on RAG-ko.
* **Promoting borderline hits (+0.06 to +0.10).** Among queries
  already hitting at rank 4-10, removing the question-form noise
  frees up the topic nouns to push the correct document to rank 1.
  On Allganize RAG-ko the fraction of queries hitting at rank 1
  rose from 37.5 % to 49.5 %.

This is a rare case where a two-line change dominates several
downstream components. It is specific to the Kiwi analyzer's POS
tag set; porting to MeCab-ko [@kudo2004mecab] would require
re-identifying the analogue noise tags.

# 5. Comparison with published numbers

We do not claim head-to-head IR supremacy over larger systems.
Instead we catalogue what competing systems report themselves and
position Synaptic on the Pareto frontier of accuracy-per-index-cost.

| System | Benchmark | Self-reported | Index-time LLM calls |
|--------|-----------|---------------|----------------------|
| Mem0 [@chhikara2025mem0] | LoCoMo | 91.6 (blend) | 1+ per turn |
| Zep [@rasmussen2025zep] | LoCoMo | 58.44 ± 0.20 (corrected) | 1+ per turn |
| HippoRAG2 [@gutierrez2025hipporag2] | MuSiQue F1 | 51.9 | 1+ per doc |
| HippoRAG2 [@gutierrez2025hipporag2] | HotpotQA str. acc. | 56.7 | 1+ per doc |
| HippoRAG2 [@gutierrez2025hipporag2] | 2Wiki R@5 | 0.904 | 1+ per doc |
| LightRAG [@guo2024lightrag] | UltraDomain win rate | 80 % (orig.) / ≈40 % (unbiased) | 1+ per doc |
| Cognee [@cognee2024] | HotPotQA (self) | 0.93 | 1+ per doc |
| **Synaptic v0.16.0, embedder-free** | **Allganize RAG-ko MRR** | **0.947** | **0** |
| **Synaptic v0.16.0, embedder-free** | **HotPotQA-dev MRR@10** (500 q) | **0.784** (Hit@10 91.8 %) | **0** |
| **Synaptic v0.16.0, embedder-free** | **2Wiki-dev MRR@10** (500 q) | **0.795** (Hit@10 91.2 %) | **0** |
| **Synaptic v0.16.0, embedder-free** | **MuSiQue-dev MRR@10** (500 q) | 0.590 (Hit@10 76.2 %) | **0** |

Two of our numbers put the LLM-free story in a concrete form:

* **HotPotQA dev** (500 q, 66,635 passage corpus). Synaptic's
  embedder-free MRR@10 of 0.784 sits above HippoRAG2's 56.7 %
  end-to-end string accuracy on the same corpus — not a like-for-
  like comparison (retrieval vs. answer), but a floor we reach
  without any LLM call during indexing.
* **2WikiMultihopQA dev** (500 q). R@5 0.501 vs. HippoRAG2's 0.904.
  Here the LLM-built entity-and-relation graph visibly helps on
  extra-long 2-hop chains; a head-to-head with Synaptic + embedder +
  cross-encoder is tracked for v0.16.1.
* **MuSiQue-Ans dev** (500 q). R@5 0.379 vs. HippoRAG2's 0.747 — by
  far the hardest dataset for a system without semantic embeddings,
  because MuSiQue's 2-4 hop chain requires bridging passages that
  share no lexical overlap.

Cross-system comparison is made harder by differing benchmark
definitions, but three observations hold:

1. Every system above Synaptic spends one or more LLM calls per
   indexed document. For a 10 k-document corpus at $10^{-3}$
   USD/call this is $\ge 10 USD$ *per index build*, and must be
   re-paid on every schema change. Synaptic's index cost is zero.
2. The Zep LoCoMo correction
   [84 % $\to$ 58.44 %; @rasmussen2025zep] is emblematic of the
   reproducibility problem in self-reported retrieval numbers. Our
   `examples/benchmark_allganize.py` runs in under two seconds on a
   laptop; anyone can falsify our number.
3. The "LLM calls at index time" column is a design decision, not a
   technical necessity. Theorem 1 holds *because* we made that
   design decision.

# 6. Related work

**GraphRAG family.** Microsoft GraphRAG [@edge2024graphrag] and the
subsequent KG-RAG survey [@peng2024kgragsurvey] established the
LLM-at-ingest pattern as the implicit default. HippoRAG
[@gutierrez2024hipporag] introduced PPR over an OpenIE-derived KG —
we borrow the PPR retrieval stage verbatim. HippoRAG 2
[@gutierrez2025hipporag2] added continual-learning framing and
hybrid dense/sparse fusion, but did not formalise any invariance
under streaming updates. LightRAG [@guo2024lightrag] is, to our
knowledge, the only prior KG-RAG with an explicit "incremental
update algorithm" component; it does not characterise the ranking
stability of that update.

**Agent memory.** Mem0 [@chhikara2025mem0], Zep/Graphiti
[@rasmussen2025zep], and MemGPT/Letta [@packer2023memgpt] address
the orthogonal problem of conversational memory. Evaluations using
LoCoMo [@maharana2024locomo] and LongMemEval [@wu2025longmemeval]
are the closest the agent-memory line comes to measuring update
robustness, but they measure *end-to-end answer accuracy after
update*, not the intrinsic top-$k$ invariance of the retriever.

**Streaming indexes.** The vector-database line
[@wang2021milvus; @guo2022manu; @singh2021freshdiskann;
@xu2023spfresh] studies streaming *at the index structure level* —
how to keep an HNSW graph or LSM segment tree consistent under
inserts and deletes. Our Theorem 1 sits above their abstraction:
given any such index is internally correct, and given BM25-shaped
scoring, the retrieval ranking is invariant across schedules.
CDC infrastructure work [@das2012databus; @carbone2015flink] gave
us the engineering template for `sync_from_database`.

**Korean retrieval.** Kiwi [@lee2024kiwi] and MeCab-ko
[@kudo2004mecab] dominate the Korean morphology landscape. The
AutoRAG benchmark [@kim2024autorag] runs tokenizer ablations but
(to our knowledge) has not studied the query-time vs. index-time
asymmetry that we exploit in Section 4. KLUE
[@park2021klue] and Allganize's public RAG-Eval
[@allganize2024rageval] supply the evaluation scaffolding we use.

**Incremental PageRank.** Ohsaka et al.'s evolving-network PageRank
[@ohsaka2015incremental] is the algorithmic line closest to our
PPR component under corpus growth. Our contribution is not a faster
incremental algorithm but a correctness statement about the
downstream ranking.

# 7. Limitations

* **Not a state-of-the-art accuracy paper.** With embedder +
  cross-encoder reranker the same pipeline reaches MRR 0.905 on
  Allganize RAG-ko, but that requires a GPU-backed service and is
  outside the laptop-reproducible baseline we want to highlight.
* **Tie-break order.** Theorem 1 gives set equality, not order
  equality. Adding a $\phi$-ordered secondary sort in v0.16.0 would
  close the 1.5 % set-agreement gap we measured, at minor cost.
* **BM25-shaped restriction.** Learned sparse retrievers with
  corpus-dependent vocabulary drift are outside the theorem's
  scope.
* **Korean-centric Section 4.** The query-stripping hyperparameters
  (the frozen set `_KO_QUERY_NOISE`) are language-specific. The
  methodology transfers but the parameters must be re-elicited for
  each language.
* **Benchmark coverage.** v0.16.0 adds three English multi-hop
  standards — HotPotQA-dev (full 66 k corpus), MuSiQue-Ans-dev, and
  2WikiMultiHopQA-dev — run at a 500-query subset each (see
  [synaptic_results.md §Tier 1.5](../comparison/synaptic_results.md)).
  Full-dataset runs and a head-to-head with Mem0 / Cognee on
  [`examples/benchmark_vs_competitors/`](../../examples/benchmark_vs_competitors/)
  are planned for v0.16.1 after the PPR stage's ``O(corpus_size)``
  first-hit is optimised. An agent-memory benchmark
  (LoCoMo or LongMemEval) is planned for v0.17.0.

# 8. Conclusion

We argued, and proved under stated assumptions, that a KG-RAG
system can be *more* reproducible under streaming updates by
*refusing* to call an LLM at index time. Theorem 1 is a structural
consequence of deterministic node IDs and idempotent upsert — both
cheap engineering choices, both impossible to guarantee when a
non-deterministic LLM mediates graph construction. The query-time
Korean ablation is a smaller but operationally meaningful gain that
shows the value of separating index-time and query-time text
normalisation.

The full Synaptic codebase, benchmark adapters, and rerun scripts
are MIT-licensed at
`github.com/PlateerLab/synaptic-memory`; every number in this paper
is regenerable from a clean clone in under three minutes of CPU
time.

# Appendix A — Proof of Corollary 1

Let $Q_{\text{clean}} = Q \setminus Q_{\text{tie}}$. For
$q \in Q_{\text{clean}}$, the first relevant document's rank is
invariant across schedules by Theorem 1 (no ties at the relevant
rank). For $q \in Q_{\text{tie}}$, the first relevant document
sits at a tie, so its rank can shift by at most one position
within the tied group. The reciprocal rank therefore changes by at
most $1/r - 1/(r+1) \le 1/r^2 \le 1/k^2$ per query, so in absolute
value no more than $1/k$. Dividing the total by $|Q|$ gives the
bound. Equality when every tied query is at rank $k-1$ or $k$.
$\square$

# Appendix B — Reproducing every number

```
git clone https://github.com/PlateerLab/synaptic-memory
cd synaptic-memory
uv sync --extra all

# Table (Section 3.4) — streaming invariance
python examples/ablation/streaming_experiment.py

# Table (Section 4.3) — Korean ablation
python examples/ablation/run_ablation.py

# Headline numbers (Abstract, Section 5)
python examples/benchmark_allganize.py
```

Each script writes a JSON / Markdown report under
`examples/ablation/diagnostics/` or
`examples/benchmark_vs_competitors/results/`, which is what was
pasted into the tables above.
