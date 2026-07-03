# Adoption Guide

This guide is for developers adding Synaptic Memory to an existing RAG or
agent application.

## Fast Smoke

```bash
pip install synaptic-memory
synaptic-quickstart
```

The default quickstart uses an in-memory graph and has no optional dependency.
Persist the sample graph with SQLite:

```bash
pip install "synaptic-memory[sqlite]"
synaptic-quickstart --db quickstart.db
```

## Recommended Installs

```bash
pip install "synaptic-memory[sqlite,korean,vector]"
pip install "synaptic-memory[mcp]"
pip install "synaptic-memory[langchain]"
pip install "synaptic-memory[all]"
```

Use the narrowest extra set that matches your deployment. Core install stays
zero-dependency for tests and in-memory prototypes.

## Python API

```python
from synaptic import SynapticGraph

graph = await SynapticGraph.from_data("./docs/", preset="rag")
try:
    result = await graph.search("refund policy exception")
finally:
    await graph.close()
```

For pre-chunked data:

```python
graph = await SynapticGraph.from_chunks(
    [{"content": "...", "title": "Manual", "source": "manual.pdf"}],
    preset="local",
)
```

## Presets

| Preset | Use when | Behavior |
|---|---|---|
| `local` | tests, local smoke, deterministic baseline | no external endpoints |
| `rag` | normal RAG app | reads `SYNAPTIC_EMBED_URL` / `SYNAPTIC_RERANK_URL` |
| `agent` | multi-turn graph exploration | `rag` plus deterministic component bridging |
| `scale` | large ingest with external indexing plan | endpoint-aware, index/router configured separately |

Use `GraphBuildOptions` when the same configuration should be reused across
`from_data()`, `from_chunks()`, and `from_database()`.

```python
from synaptic import GraphBuildOptions, SynapticGraph

options = GraphBuildOptions.rag().with_overrides(connect=True)
graph = await SynapticGraph.from_data("./docs/", options=options)
```

## Optional MCP Server

```bash
pip install "synaptic-memory[mcp]"
synaptic-mcp --db knowledge.db
```

Without the `mcp` extra, `synaptic-mcp --help` still works and prints the
install command instead of failing with a Python traceback.
