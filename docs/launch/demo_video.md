# Demo Video Plan

Goal: a 60-75 second terminal-first video that makes Synaptic Memory's position
obvious without requiring a long explanation.

Core message:

> Synaptic Memory is not a vector database replacement. It is the graph, tool,
> and memory layer around documents, SQL rows, embeddings, and RAG agents.

---

## Recommended Video

Title:

> Synaptic Memory in 60 seconds

Recording command:

```bash
uv run python examples/launch_demo.py
```

Fast verification command:

```bash
uv run python examples/launch_demo.py --no-pause
```

Optional recording tools:

```bash
# Terminal recording
asciinema rec docs/launch/synaptic-memory.cast \
  -c 'uv run python examples/launch_demo.py'

# GIF export if agg is installed
agg docs/launch/synaptic-memory.cast docs/launch/synaptic-memory.gif
```

GUI screen recording also works. Use a large terminal, 120 columns, and a dark
theme. Zoom the terminal enough that text is readable on mobile.

---

## Shot List

| Time | Screen | Voiceover / caption |
|------|--------|---------------------|
| 0-5s | Title lines | "Synaptic Memory is graph memory for RAG agents." |
| 5-15s | Build graph | "It indexes documents and structured rows without LLM calls by default." |
| 15-35s | Search results | "A query can hit both policy text and support-ticket rows." |
| 35-50s | Feedback | "When evidence helps, record feedback instead of losing that signal." |
| 50-65s | Health metadata | "Events, scores, and health signals live as metadata, not prompt bulk." |
| 65-75s | Closing line | "Start local, then swap backend to PostgreSQL, Kuzu, Qdrant, or MinIO." |

---

## Exact Caption Copy

Use these short captions if editing the video:

1. "LLM-free indexing by default"
2. "Documents + SQL rows become one graph"
3. "Search returns evidence, not just chunks"
4. "Feedback becomes memory metadata"
5. "Local first. Infra-ready when you grow."

---

## Social Post With Video

```text
Synaptic Memory in 60 seconds:

- build a graph from docs + SQL rows
- search with agent-ready evidence
- record feedback when evidence helps
- inspect memory health metadata
- start local, swap to PostgreSQL/Kuzu/Qdrant later

https://github.com/PlateerLab/synaptic-memory
```

---

## Production Notes

- Keep the recording under 75 seconds.
- Do not show API keys or local private paths.
- Prefer the `--no-pause` run for CI checks, and the default paused run for
  screen recording.
- Avoid benchmark claims in the video. Link reports in the post text instead.
