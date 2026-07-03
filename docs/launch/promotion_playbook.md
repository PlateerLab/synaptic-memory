# Promotion Playbook

This is the working launch plan for Synaptic Memory. Keep claims tied to the
current README, reports, and reproducible commands.

---

## Positioning

Short form:

> Synaptic Memory is a Python knowledge graph + MCP tool server for agentic RAG:
> deterministic indexing by default, infra-friendly backends, live DB sync, and
> optional memory feedback.

One-line GitHub/social variant:

> Graph memory for RAG agents: documents + SQL rows + edges + MCP tools, with
> LLM-free indexing by default.

What to emphasize:

- LLM-free deterministic default path.
- Mixed structured and unstructured retrieval.
- MCP tools for multi-turn agent search.
- Live database sync.
- Infra options: SQLite locally, PostgreSQL for shared service, Kuzu/Qdrant/MinIO
  for scale-out composition.
- Memory operating layer: feedback, provenance, scope-aware scores, health
  signals.

What not to claim:

- Do not call it a vector database replacement.
- Do not say it is 10TB-ready out of the box.
- Do not imply the memory layer decides truth automatically.
- Do not quote benchmark numbers without linking the report and reproduction
  path.

---

## Launch Assets

Must-have:

- GitHub README with quickstart and "Why not just RAG?"
- PyPI metadata links: repository, docs, changelog, issues, discussions, reports,
  quickstart.
- GitHub topics and description.
- GitHub release notes.
- 60-second generated demo and recording command: `docs/launch/demo_video.md`.
- Show HN draft: `docs/launch/show_hn_draft.md`.

Nice-to-have:

- Edited voiceover version of the generated terminal GIF/MP4.
- Architecture image showing documents/SQL -> graph -> EvidenceSearch -> MCP
  agent tools.
- Short Korean post for GeekNews / LinkedIn.

---

## GitHub Release Draft

Title:

> Synaptic Memory: graph memory layer for agentic RAG

Body:

~~~markdown
Synaptic Memory is a Python knowledge graph + MCP tool server for agentic RAG.
The default path is deterministic and LLM-free at indexing time, while optional
layers add embeddings, reranking, OpenIE extraction, retrieval feedback, and
memory health signals.

Highlights:

- `synaptic-quickstart` for a 5-minute local smoke test.
- `examples/launch_demo.py` for a 60-second launch video over documents, SQL
  rows, retrieval feedback, and memory health metadata.
- `SynapticGraph.from_data()`, `from_chunks()`, and `from_database()` presets:
  `local`, `rag`, `agent`, `scale`.
- Infrastructure paths for SQLite, PostgreSQL + pgvector, Kuzu, Qdrant, and
  MinIO/S3-compatible blob storage.
- MCP tools for deep search, comparison search, structured filters, aggregation,
  FK joins, top-N queries, ingestion, and CDC sync.
- Memory operating layer: retrieval events, feedback, provenance, scope-aware
  reinforcement, and health reporting.

Try it:

```bash
pip install "synaptic-memory[sqlite,korean,vector]"
synaptic-quickstart --db quickstart.db
```

Reports:

- RAG comparison: docs/REPORT-rag-vs-synaptic.md
- Memory operating layer eval: docs/REPORT-memory-operating-layer-eval.md
- Reproducible results log: docs/comparison/synaptic_results.md
~~~

---

## Social Copy

### X / Twitter

```text
I built Synaptic Memory: a Python knowledge graph + MCP server for agentic RAG.

Default path:
- no LLM calls at indexing time
- local SQLite quickstart
- optional PostgreSQL/Kuzu/Qdrant/MinIO infra
- MCP tools for multi-turn search
- feedback + memory health signals

https://github.com/PlateerLab/synaptic-memory
```

### LinkedIn

```text
I have been building Synaptic Memory, a Python library for teams that need RAG
over both documents and SQL data without turning every indexing job into an LLM
job.

It builds a deterministic knowledge graph by default, exposes MCP tools for
agentic multi-turn search, supports live database sync, and can run locally on
SQLite or connect to PostgreSQL, Kuzu, Qdrant, and MinIO/S3-style storage.

The goal is not to replace vector databases. It is to provide the graph, tool,
and memory layer around existing documents, SQL data, embedding endpoints, and
LLM agents.

GitHub: https://github.com/PlateerLab/synaptic-memory
PyPI: https://pypi.org/project/synaptic-memory/
```

### Reddit

Use only in communities where project posts are allowed. Lead with the problem,
not the link.

```text
I built a Python knowledge graph + MCP server for RAG agents over mixed
documents and SQL data.

The main design choice: indexing is deterministic by default, so adding data
does not require an LLM call. The agent can still use embeddings/rerankers and
MCP tools at query time.

Useful if you care about local/on-prem deployments, live DB sync, Korean FTS, or
agent search over both structured and unstructured data.

I would appreciate feedback on the tool surface and benchmark plan:
https://github.com/PlateerLab/synaptic-memory
```

---

## Channel Order

1. GitHub release.
2. PyPI release metadata check.
3. Record the 60-second terminal demo from `docs/launch/demo_video.md`.
4. HN Show HN.
5. Korean mirror post: GeekNews / LinkedIn.
6. Reddit posts only where allowed.
7. Follow-up technical blog post:
   "Why RAG agents need graph tools, not only vector search."

---

## Follow-Up Blog Outline

Title:

> Why RAG agents need graph tools, not only vector search

Outline:

1. Plain RAG is strong but chunk-local.
2. Enterprise corpora mix documents, tables, foreign keys, policies, and updates.
3. Index-time LLM extraction is useful but costly and hard for on-prem.
4. Deterministic graph first: documents, chunks, rows, references, FK edges.
5. Query-time agent tools: search, expand, filter, aggregate, join, top-N.
6. Memory operating layer: feedback and provenance as metadata, not prompt bulk.
7. Honest limits: not web-scale without an operating layer, not truth discovery.

---

## Measurement Backlog

Good promotion gets stronger when each claim has a command behind it.

- Clean-install quickstart timing.
- Public Korean FTS benchmark.
- Agent search benchmark with DeepSeek Flash profile.
- Postgres CDC parity smoke.
- CompositeBackend smoke with Kuzu + Qdrant + MinIO.
- Large-corpus benchmark report linked from `docs/comparison/synaptic_results.md`.
