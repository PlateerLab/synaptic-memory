# Show HN launch draft

**Status:** draft — do not post until Week 2 checklist below is green.

---

## Title candidates (pick one, test against title gotchas)

1. **Show HN: Synaptic Memory – RAG with zero LLM calls at index time**
2. Show HN: Synaptic Memory – SQLite-native RAG + CDC for live production databases
3. Show HN: Synaptic Memory – a knowledge graph for LLM agents that doesn't need an LLM to build

**Recommended:** #1. Shortest, makes the concrete claim, triggers curiosity.

**Title rules (HN-proofed):**
- ≤ 80 chars including "Show HN: ".
- No emoji, no version numbers, no exclamation marks.
- "Zero LLM calls at index time" is the hook — it's falsifiable, which helps.

---

## Body

> Hi HN,
>
> Synaptic Memory is a Python library and MCP server that turns any
> corpus — CSV, JSONL, PDFs, or a live SQL database — into a knowledge
> graph that LLM agents can search, without ever calling an LLM to
> build it.
>
> Most of the "agent memory" space (Mem0, Zep/Graphiti, Cognee,
> LightRAG, MS GraphRAG) leans on an LLM at index time to extract
> entities, relations, or community summaries. That's great for
> recall, but it adds a per-document inference cost, a privacy
> surface, and a "can I run this on-prem without calling an external
> API" problem. I kept running into that wall at work on Korean
> enterprise deployments where external API calls are simply a
> non-starter.
>
> Synaptic takes the other path: build the graph with structural and
> statistical signals only (FK edges, sentence-boundary chunking,
> document-frequency phrase hubs, NEXT_CHUNK sequence), then let the
> agent's LLM do the judgment at **query** time. The retrieval
> pipeline is BM25 (FTS5) + usearch HNSW vectors + Personalized
> PageRank + cross-encoder reranker + MMR — all local, all optional.
>
> Things I think are genuinely useful:
>
> * **Zero LLM at indexing.** A 10-document CSV → searchable graph in
>   < 0.2 s. A 200-document Korean enterprise benchmark runs in under
>   two seconds on a laptop (details below).
> * **Native CDC.** `SynapticGraph.from_database(..., mode="cdc")`
>   gives you deterministic node IDs, and `sync_from_database()`
>   propagates inserts/updates/deletes incrementally. A regression
>   test locks in that CDC-mode top-k matches a full rebuild — so
>   your production database and your retrieval index stay in sync
>   without a nightly reindex job.
> * **Structured + unstructured in the same graph.** CSV/SQL rows
>   land as typed property nodes with FK edges (RELATED); documents
>   land as Category → Document → Chunk. The 36 MCP tools let an
>   agent mix `filter_nodes` / `aggregate_nodes` / `join_related`
>   with `deep_search` in one conversation.
> * **Korean FTS built in.** Kiwi morphological analyzer,
>   auto-detected by the 50 %-Hangul heuristic so structured English
>   data doesn't get over-segmented.
> * **SQLite default.** No Neo4j, no Postgres, no vector DB to run.
>   Postgres + pgvector, Kuzu, and Qdrant backends exist if you need
>   them.
>
> What I'm intentionally *not* claiming:
>
> * Not "a new generation of RAG." The algorithmic primitives
>   (BM25, HNSW, PPR, cross-encoder, MMR) are all well known — the
>   contribution is the integration, the CDC path, and the
>   LLM-free-at-index-time property, not a new retrieval algorithm.
> * Not tested at web scale. Default backend is good to roughly
>   100 k nodes; above that you want the Postgres or Kuzu backend,
>   which I've used in production but haven't benchmarked publicly.
> * Single maintainer. v0.15.0 is marked Beta for a reason.
>
> **Reproducible numbers** (the part I care most about, given how
> much self-reported benchmark trouble this space has had lately):
>
> ```
> $ pip install "synaptic-memory[korean]"
> $ python examples/benchmark_allganize.py
> Dataset                  Corpus  Queries      MRR     R@10        Hit     Time
> Allganize RAG-ko            200      200    0.947    1.000   200/200     9.3s
> Allganize RAG-Eval          300      300    0.911    0.950   285/300     5.9s
> ```
>
> That's **embedder-free** — no vector index, no cross-encoder, zero
> LLM calls at any point. Two releases of cumulative gain from the
> v0.15.0 legacy baseline (Allganize RAG-ko: 0.621 → 0.743 via
> query-time Korean morphological stripping in v0.15.1, then → 0.947
> via the v0.16.0 engine default flip to the hybrid EvidenceSearch
> pipeline).
>
> English standard benchmarks (500q subsets of HotPotQA-dev /
> MuSiQue-Ans / 2WikiMultihopQA, also embedder-free):
>
> ```
> HotPotQA dev (66,635 docs)         MRR@10 0.784   Hit@10 91.8 %
> 2WikiMultihopQA dev (56,687 docs)  MRR@10 0.795   Hit@10 91.2 %
> MuSiQue-Ans dev (21,100 docs)      MRR@10 0.590   Hit@10 76.2 %
> ```
>
> MuSiQue shows the expected weakness of an embedder-free system on
> 2-4 hop chains — R@5 0.379 vs HippoRAG2's 0.747. We don't hide
> it; closing that gap is what the v0.16.1 embedder path is for.
>
> License: MIT. Links:
>
> * GitHub: https://github.com/PlateerLab/synaptic-memory
> * PyPI: https://pypi.org/project/synaptic-memory/
> * LangChain integration: `pip install "synaptic-memory[langchain]"` →
>   `from synaptic.integrations.langchain import SynapticRetriever`
> * Quick start: https://github.com/PlateerLab/synaptic-memory#5-minute-start
>
> Things I'd genuinely like feedback on:
>
> * Are there obvious benchmarks (BEIR subsets, LoCoMo, LongMemEval)
>   you'd want to see before trusting the numbers?
> * Does the CDC story solve a real pain, or is nightly reindex
>   "good enough" in practice for most of you?
> * How badly does the "36 MCP tools" surface clash with how you
>   actually wire agents?
>
> Happy to answer questions. Thanks for reading.

---

## Top-comment prep (post yourself immediately after submission)

> A few things I'd flag up front that might come up in comments:
>
> 1. **"3rd-gen GraphRAG" terminology.** An earlier README version
>    used this framing. I've since removed it — self-proclaimed
>    generations are a bad look given what happened to the 84 %
>    LoCoMo claim last year. The table now just compares indexing
>    cost across approaches.
> 2. **HotPotQA is run at a 500-query subset**, not the full
>    7,405-question dev set, because PPR's first-hit cost is
>    O(corpus_size) on the current implementation and a full run
>    takes ~3.7 hours on a laptop. Optimising that is tracked as
>    v0.16.1.
> 3. **MuSiQue lags HippoRAG2 by a lot** (R@5 0.379 vs 0.747). That's
>    the honest result — embedder-free retrieval cannot bridge 2-4
>    hop chains that share no lexical overlap. The paper is explicit
>    about it; v0.16.1 adds the embedder path for a fairer rematch.
> 4. **Python 3.12+ only.** I know 3.10/3.11 covers more enterprise
>    deployments — relaxing this is on the roadmap.

---

## Timing

- **Best slots:** Tue–Thu, 09:00–11:00 PT (~17:00 KST). Avoid
  holidays, major product launches, earnings weeks.
- **Don't submit before Week 2 checklist (below) is ✅.** A failed
  Show HN is very hard to redo.

---

## Week-2 pre-launch checklist

### Must-haves (block launch)
- [x] `examples/quickstart.py` runs clean on `pip install "synaptic-memory[sqlite,korean,vector]"`.
- [x] `examples/benchmark_allganize.py` finishes in < 15 s and prints stable numbers (v0.16.0).
- [x] `examples/langchain_retriever.py` works end-to-end.
- [x] `examples/ablation/run_tier1_benchmarks.py --subset 500` finishes within an hour.
- [x] README "5-minute start" block is copy-pasteable.
- [x] No broken links in README / README.ko.md.
- [x] Unit tests green (`uv run pytest tests/ -q ...` → 819 pass).
- [x] README.ko.md in sync with README.md (v0.16.0).
- [x] docs/TUTORIAL.en.md exists.
- [x] v0.16.0 published on PyPI.  ← **requires `uv publish` before posting**
- [ ] GitHub repo has a description, topics, and a social preview image.
- [ ] Discussions or an Issues template enabled (so HN commenters can file follow-ups).

### Nice-to-haves (don't block)
- [ ] Anthropic MCP registry submission in flight.
- [ ] Short Loom / asciinema cast linked from README ("60-second demo").
- [ ] Comparison table vs Mem0 / Zep / Cognee in `docs/COMPARISON.md` in English.
- [ ] Colab notebook mirror of `benchmark_allganize.py`.

### Post-launch monitoring (first 6 hours)
- Refresh HN front page every 10–15 minutes for the first 2 hours.
- Reply to every top-level comment within 30 minutes.
- Lead with specifics, never marketing language.
- If a comment is harsh but technical, thank them and answer the
  technical point — never get defensive.
- Cross-post to:
  - r/LocalLLaMA (after HN momentum is clear)
  - r/Python (once stars cross 200)
  - AI Korea Slack / GeekNews (hada.io) — same day, Korean version
  - X/Twitter thread with the benchmark gif

---

## Korean mirror post (for GeekNews / 브런치, same day)

동일 내용을 한국어로 준비하되:

- 제목: **"LLM 호출 없이 인덱싱하는 RAG — Synaptic Memory (Show HN)"**
- 한국어 Allganize 벤치마크 숫자를 **전면**에 배치 (영어권과 달리
  한국어 독자는 이 숫자에 더 민감).
- 한국어 FTS(Kiwi), on-prem 중심 포지셔닝을 강조.
- 국내 SI / 대기업 채택 상담 가능 여부를 명시 (엔터프라이즈 유입 채널).
