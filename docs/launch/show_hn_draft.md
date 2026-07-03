# Show HN Draft

**Status:** draft. Do not post until the launch checklist is green and the
current release is on PyPI.

---

## Title Candidates

1. **Show HN: Synaptic Memory - a graph memory layer for agentic RAG**
2. Show HN: Synaptic Memory - LLM-free indexing for agent search
3. Show HN: Synaptic Memory - knowledge graph + MCP tools for RAG agents

Recommended: #1. It is short, accurate, and does not over-index on a benchmark
claim.

---

## Body Draft

Hi HN,

I built Synaptic Memory, a Python library and MCP server that turns documents,
CSV/JSONL files, and SQL databases into a searchable knowledge graph for LLM
agents.

The design goal is simple: keep indexing deterministic by default, then give
the agent better tools at query time. A lot of graph/RAG systems use an LLM at
index time to extract entities, relations, or summaries. That can work well, but
it also creates a per-document cost, a privacy surface, and an on-prem deployment
problem. Synaptic's default path uses structural and statistical signals instead:
document hierarchy, chunks, foreign keys, explicit references, phrase/entity
hubs, graph expansion, BM25, optional vectors, optional reranking, and MCP tools.

What it does:

- Ingests directories, CSV/JSONL, office files through an optional parser, and
  relational databases.
- Builds a graph of documents, chunks, table rows, and typed edges.
- Supports live DB sync via CDC-style incremental updates for practical
  database-backed corpora.
- Exposes MCP tools such as `deep_search`, `compare_search`, `filter_nodes`,
  `aggregate_nodes`, `join_related`, and `top_nodes`.
- Supports local SQLite, PostgreSQL + pgvector, Kuzu, Qdrant, and MinIO/S3-style
  composition through optional extras.
- Adds an opt-in memory operating layer: retrieval events, feedback,
  provenance, scope-aware reinforcement, and health signals.

Five-minute local test:

```bash
pip install "synaptic-memory[sqlite,korean,vector]"
synaptic-quickstart --db quickstart.db
```

Basic usage:

```python
from synaptic import SynapticGraph

graph = await SynapticGraph.from_data("./docs/", preset="rag")
result = await graph.search("refund exception")
```

What I am not claiming:

- Not a vector database replacement. It is the graph/tool/memory layer around
  your documents, SQL data, embedding endpoint, and agent runtime.
- Not web-scale out of the box. The core library provides backend contracts and
  retrieval logic; multi-terabyte production deployments still need ingestion
  workers, external indexes, ACL filtering, monitoring, and backup/restore.
- Not magic truth discovery. Feedback and health signals are observations, not
  automatic truth judgments.

Links:

- GitHub: https://github.com/PlateerLab/synaptic-memory
- PyPI: https://pypi.org/project/synaptic-memory/
- Quick start: https://github.com/PlateerLab/synaptic-memory#5-minute-start
- RAG comparison report: https://github.com/PlateerLab/synaptic-memory/blob/main/docs/REPORT-rag-vs-synaptic.md
- Memory operating layer report: https://github.com/PlateerLab/synaptic-memory/blob/main/docs/REPORT-memory-operating-layer-eval.md

License: Apache-2.0. Python 3.12+.

I would especially like feedback on:

- Whether the MCP tool surface is too broad or useful in real agent setups.
- Which public RAG / agent-search benchmarks would make the results more
  trustworthy.
- Whether CDC-style database sync solves a real workflow pain, or whether most
  teams are fine with periodic rebuilds.

Thanks for reading.

---

## First Comment Prep

Post this as the first comment if the HN thread starts getting technical
questions:

> A few clarifications:
>
> - The default path is LLM-free at indexing time. OpenIE/LLM extraction exists
>   as an opt-in layer, not the default.
> - Qdrant and MinIO are helper services behind `CompositeBackend`; they do not
>   store the full graph by themselves.
> - The memory layer records events, feedback, provenance, and health signals as
>   metadata. It does not append raw provenance into `Node.content`, and it does
>   not dump raw ledger rows into prompts.
> - The project is beta. The strongest current use case is infra-friendly RAG for
>   mixed document + SQL corpora, especially when an agent needs multi-turn
>   search tools rather than one-shot vector retrieval.

---

## Launch Checklist

Blockers:

- [ ] Latest README and README.ko.md are merged to `main`.
- [ ] PyPI release is current.
- [ ] `synaptic-quickstart --json` works from a clean install.
- [ ] GitHub repo description, topics, homepage, and social preview are set.
- [ ] Issues are enabled.
- [ ] Discussions are enabled or the draft avoids promising Discussions.
- [ ] Release notes link to the current reports and quickstart.

Nice-to-have:

- [ ] 60-second terminal GIF or MP4 from `docs/launch/demo_video.md`.
- [ ] One architecture image for social posts.
- [ ] Korean mirror post for GeekNews / LinkedIn.
- [ ] A small "RAG vs Synaptic" diagram in the docs.

Post-launch:

- Watch comments for the first 2 hours.
- Answer technical questions with specifics and links.
- Do not ask for upvotes.
- If a benchmark question is unclear, point to the reproduction command or say
  exactly what has not been measured yet.
