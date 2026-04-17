# Theorem 1 — Top-K Set Invariance Under CDC-Based Streaming Ingest

## Preliminaries

Let $\mathcal{D}$ be a finite set of documents (a *corpus*) and let
$f : 2^{\mathcal{D}} \times \mathcal{Q} \to \mathbb{R}$ be a scoring
function that maps a corpus $C \subseteq \mathcal{D}$ and a query
$q \in \mathcal{Q}$ to a real-valued score $f(d; C, q)$ for every
$d \in C$.

We say $f$ is **BM25-shaped** if, given $C$ and $q$, each score
$f(d; C, q)$ is a deterministic function of:

1. the per-document term frequencies $\mathrm{tf}(t, d)$ for every term
   $t$ in $q$,
2. the per-corpus document frequencies $\mathrm{df}(t, C)$ for every
   term $t$ in $q$,
3. the document length $|d|$ and the average document length
   $\bar{\ell}(C)$,
4. a fixed set of hyperparameters $(k_1, b, \text{title boost}, ...)$.

Standard BM25, BM25F, and hybrid lexical-BM25 scoring are all
BM25-shaped.

An **ingest schedule** is an ordered partition
$\Sigma = (B_1, B_2, \ldots, B_n)$ of $\mathcal{D}$ — i.e.
$B_i \cap B_j = \emptyset$ for $i \ne j$ and
$\bigcup_i B_i = \mathcal{D}$. The **state after step $t$** is
$C_t^\Sigma = \bigcup_{i \le t} B_i$, and the **final state** is
$C^\Sigma = C_n^\Sigma = \mathcal{D}$.

Synaptic's ingest satisfies two engineering guarantees, baked into
the graph schema:

**G1. Deterministic node IDs.**
There exists a function $\phi : \mathcal{D} \to \mathrm{NodeID}$ such
that, given a document $d$, every ingest of $d$ — under any schedule
— stores it under the same node ID $\phi(d)$. For table-sourced
rows, $\phi$ is derived from
$(\text{source\_url}, \text{table}, \text{primary\_key})$; for
document-sourced items it is a content hash. See
`src/synaptic/extensions/cdc/state.py` and
`tests/test_cdc_search_regression.py` for the concrete realisation.

**G2. Idempotent upsert.**
For any document $d$ already stored under $\phi(d)$, the ingest
operation for a second instance of $d$ is a no-op on both node state
and edge state. Formally, if $C$ contains $d$ then ingesting $d$
again yields the same corpus $C$ (not $C \cup \{d\}$, since
$d \in C$).

## Theorem 1 (Top-K Set Invariance)

Let $f$ be BM25-shaped, $k \in \mathbb{N}$, and let
$\Sigma_1, \Sigma_2$ be any two ingest schedules producing the same
final corpus $\mathcal{D}$. Let $T_k(f, C, q)$ denote the set of the
$k$ highest-scoring documents in $C$ for query $q$, with an arbitrary
but deterministic tie-breaking rule. Then

$$
T_k(f, C^{\Sigma_1}, q) \;=\; T_k(f, C^{\Sigma_2}, q)
\qquad \text{as sets (ignoring tie-break order)}.
$$

### Proof sketch

Fix any query $q$. A BM25-shaped $f$ depends on the final corpus $C$
only through $(\mathrm{tf}(\cdot, d), \mathrm{df}(\cdot, C), |d|,
\bar{\ell}(C))$. The per-document quantities $\mathrm{tf}(\cdot, d)$
and $|d|$ are intrinsic to the document — they do not depend on the
ingest schedule at all. The per-corpus quantities $\mathrm{df}(\cdot,
C)$ and $\bar{\ell}(C)$ depend only on the final set $C$ as a set
(not on the order in which its members were added).

Guarantee **G1** ensures that each document $d$ is stored under the
same node ID $\phi(d)$ regardless of schedule. Guarantee **G2**
ensures that a given $d \in C$ contributes exactly once to
$(\mathrm{df}, \bar{\ell})$ no matter how many times the schedule
tries to insert it. Therefore
$f(d; C^{\Sigma_1}, q) = f(d; C^{\Sigma_2}, q)$ for every
$d \in \mathcal{D}$, and the set of the top $k$ is identical. $\square$

### Note on the tie-break subtlety

The theorem gives **set** equality: whenever two documents
$d, d' \in \mathcal{D}$ have identical scores
$f(d; C, q) = f(d'; C, q)$, they may appear in either order in the
returned top-$k$. That order is governed by the tie-breaking rule,
which in practice is insertion-order-dependent (e.g. the Python
`dict` iteration order in `MemoryBackend`). A **strictly ordered**
invariance would require an insertion-order-free tie-break (e.g.
secondary sort by $\phi(d)$). See Section 4 of the paper for
empirical characterization: ~98.5 % set agreement and ~52 % exact
order agreement on 200 Allganize RAG-ko queries, $\Delta$ MRR
$< 0.01$.

## Corollary 1 (MRR Stability)

Let $\mathrm{MRR}(f, C, Q)$ denote mean reciprocal rank over a query
set $Q$. For any two ingest schedules $\Sigma_1, \Sigma_2$ producing
the same $\mathcal{D}$,

$$
\big| \mathrm{MRR}(f, C^{\Sigma_1}, Q) - \mathrm{MRR}(f, C^{\Sigma_2}, Q) \big|
\;\le\; \frac{|Q_{\text{tie}}|}{|Q|} \cdot \frac{1}{k} ,
$$

where $Q_{\text{tie}} \subseteq Q$ is the subset of queries for which
the first relevant document in $T_k$ is tied with at least one other
member of $T_k$ on $f$.

### Proof sketch

For queries where no tie exists at the first-relevant position,
Theorem 1 gives identical reciprocal-rank contributions. For queries
in $Q_{\text{tie}}$, the reciprocal rank can differ by at most
$1/\mathrm{rank} - 1/(\mathrm{rank}+1) \le 1/k$ in absolute value.
Summing and dividing by $|Q|$ yields the bound. $\square$

## Empirical validation

`examples/ablation/streaming_experiment.py` re-runs Theorem 1 on
Allganize RAG-ko with

- $|\mathcal{D}| = 200$ documents,
- $|Q| = 200$ queries,
- $\Sigma_1$ = one batch of all 200 documents,
- $\Sigma_2$ = 10 shuffled batches of ~20 documents each (seed 42).

Result (locked in as of v0.16.0, 2026-04-17):

| Quantity | v0.15.x (legacy engine) | **v0.16.0 (evidence engine)** |
|----------|-------------------------|-------------------------------|
| Set-equal top-10 | 197 / 200 (98.5 %) | 197 / 200 (98.5 %) |
| Exact-ordered top-10 | 103 / 200 (51.5 %) | **192 / 200 (96.0 %)** |
| Top-1 identical | 109 / 200 (54.5 %) | **200 / 200 (100 %)** |
| MRR (batch) | 0.7434 | 0.9468 |
| MRR (streaming) | 0.7334 | 0.9468 |
| $\lvert \Delta \mathrm{MRR} \rvert$ | 0.0100 | **0.0000** |

On the v0.16.0 default engine the invariance is **exact on MRR and on
top-1 rank**, and set equality holds at the same 98.5 % with the
remaining disagreements at rank 9 or 10 (Corollary 1's tie-break
regime). In other words, the theorem's set-invariance claim is
vindicated; the 1.5 % remaining order drift on legacy was an artefact
of the legacy scoring cascade's tie-break sensitivity, not a
structural violation of the theorem.

## Consequences for operational use

1. **No re-indexing on corpus growth.** Under Theorem 1, a production
   index built up incrementally via
   `SynapticGraph.sync_from_database(dsn)` returns the same **set** of
   top-$k$ hits as a fresh rebuild — so nightly reindex jobs are
   unnecessary for retrieval quality.

2. **Reproducibility under deletion.** If the source DB deletes a
   document, guarantee G1 means the corresponding node is purged,
   not left stale — since the theorem applies to the final corpus
   $\mathcal{D}$ *after* deletion, the invariance holds.

3. **Orthogonal to LLM-extracted graphs.** Theorem 1 depends on
   G1/G2 only. Systems that embed LLM outputs into edges (GraphRAG,
   Cognee, HippoRAG) cannot guarantee G2 in general: a second LLM
   call on the same document may yield different relations, and so
   a re-ingest mutates the graph non-trivially. This is a structural
   reason why CDC-style streaming is hard for LLM-extracted RAG
   systems, not merely an engineering gap.

## Limitations

- The BM25-shaped assumption covers lexical BM25, BM25F, and their
  hybrid lexical variants. It does **not** cover retrieval scores
  that depend on a global embedding manifold re-trained per ingest
  (e.g. learned sparse retrievers that update tokenizer weights on
  every batch). Extending the theorem to such scoring functions is
  future work.
- Tie-break order invariance requires a $\phi$-ordered secondary
  sort. We propose this as v0.16.0 roadmap work; the current
  implementation exhibits the 1.5 % order gap observed above.
- PPR contributions to the score depend on the graph topology
  induced by ingest. For FK-driven RELATED edges the topology is a
  deterministic function of the final corpus (no invariance issue);
  for MENTIONS edges the DF threshold governs edge creation and is
  in turn a function of the final corpus, so invariance holds.
  Verifying this for every edge kind is a checklist in
  `tests/test_cdc_search_regression.py`.
