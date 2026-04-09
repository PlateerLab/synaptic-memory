# Synaptic Memory

Brain-inspired knowledge graph for LLM agents and multi-agent systems.

Agents automatically structure their operational data — tool calls, decisions, outcomes, lessons — into an **auto-constructed ontology**, enabling self-retrieval and reasoning over past experiences. Library + MCP server.

[![CI](https://github.com/PlateerLab/synaptic-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/PlateerLab/synaptic-memory/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![Python](https://img.shields.io/pypi/pyversions/synaptic-memory)](https://pypi.org/project/synaptic-memory/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why

LLM agents **don't remember.** They repeat the same mistakes, fail to leverage past successes, and can't access accumulated team knowledge.

Traditional RAG stops at "chunk documents and search by vector." But agents need more than document retrieval — they need **structured experience**:

- "What decision did I make last time in this situation, and what was the outcome?"
- "Has this pattern failed before? Why?"
- "What rules should I follow when using this tool?"

Synaptic Memory borrows the answer from **how the brain works.**

---

## Differentiators

| | Synaptic Memory | Cognee | Mem0 | LightRAG |
|---|---|---|---|---|
| **Agent experience learning** | Hebbian co-activation | - | - | - |
| **Memory consolidation (4-tier)** | L0 → L1 → L2 → L3 | - | Partial | - |
| **Auto-ontology construction** | Rules + LLM + Embedding | LLM only | - | LLM only |
| **Multi-axis ranking** | relevance x importance x recency x vitality x context | - | - | - |
| **Zero-dep core** | Pure Python | - | - | - |
| **MCP server** | 16 tools | - | - | - |
| **Korean optimization** | FTS + synonym tuning | - | - | - |

### Benchmarks (FTS only, no embedding)

| Dataset | Corpus | MRR | nDCG@10 | R@10 |
|----------|--------|-----|---------|------|
| Allganize RAG-Eval (Finance/Medical/Legal) | 300 | **0.793** | 0.810 | 0.870 |
| HotPotQA-24 (multi-hop, Cognee comparison) | 226 | **0.754** | 0.636 | 0.729 |
| AutoRAGRetrieval (enterprise) | 720 | **0.639** | 0.677 | 0.800 |
| KLUE-MRC (Korean QA) | 500 | **0.607** | 0.643 | 0.760 |

---

## Design Philosophy — Four Principles from the Brain

### 1. Spreading Activation — Associative Search

When the brain hears "deploy," it co-activates CI/CD, rollback, incidents, and monitoring.

```
Search: "deployment"
  → FTS match: [CI/CD pipeline, deployment automation]
  → Neighbor activation: [rollback strategy, canary deployment, incident response rules]
  → Resonance ranking: relevance × importance × recency × vitality × context
```

### 2. Hebbian Learning — "What fires together, wires together"

```
Agent uses [PostgreSQL selection] + [vector search implementation] together → success
  → edge weight += 0.1 → co-activated in future searches

Agent uses [skip tests] + [production deploy] together → failure
  → edge weight -= 0.15 → failure experience surfaces first
```

### 3. Memory Consolidation — Keep only what matters

```
L0 (Raw, 72h)      ← All records. Deleted after 72h if not accessed.
L1 (Sprint, 90d)   ← 3+ accesses. Retained for 90 days.
L2 (Monthly, 365d) ← 10+ accesses. Retained for 1 year.
L3 (Permanent)     ← 80%+ success rate. Permanently preserved. (Demoted below 60%)
```

### 4. Auto-Ontology — Structure knowledge for future retrieval

**"When will an agent search for this knowledge?"** — metadata is auto-generated based on predicted future queries:

```python
await graph.add("Payment Outage Postmortem", "PG API timeout caused...")

# LLM auto-generates:
# kind: LESSON
# tags: ["payment", "PG", "timeout", "circuit-breaker"]
# search_keywords: ["payment failure cause", "PG outage response", "API timeout fix"]
# search_scenarios: ["searching past cases when payment system fails"]
# relations to existing nodes: --[LEARNED_FROM]--> "deployment decision"
```

Three-tier auto-construction:

| Mode | Configuration | Cost | Details |
|------|--------------|------|---------|
| **Rule-based** | `RuleBasedClassifier()` | Free | Keyword matching, zero-dep |
| **+ Embedding** | `+ RuleBasedRelationDetector()` + embedder | Free (local) | Cosine similarity auto-linking |
| **+ LLM** | `LLMClassifier()` + `LLMRelationDetector()` | Local/API | Search keyword prediction, semantic relation extraction |

---

## Install

```bash
pip install synaptic-memory                      # Core (zero deps)
pip install synaptic-memory[embedding]           # + auto-embedding (Ollama/vLLM)
pip install synaptic-memory[sqlite]              # + SQLite FTS5
pip install synaptic-memory[scale]               # Neo4j + Qdrant + MinIO + embedding
pip install synaptic-memory[mcp]                 # + MCP server
pip install synaptic-memory[all]                 # Everything
```

## Quick Start

### 1. In-memory — zero-dep, instant start

```python
from synaptic import SynapticGraph, ActivityTracker

async def main():
    graph = SynapticGraph.memory()
    tracker = ActivityTracker(graph)

    # Search past experiences (intent auto-inferred)
    result = await graph.agent_search("DB migration failure")

    # Record a decision
    session = await tracker.start_session(agent_id="my-agent")
    decision = await tracker.record_decision(
        session.id,
        title="Choose PostgreSQL",
        rationale="Need vector search + ACID",
        alternatives=["MongoDB", "SQLite"],
    )

    # Record outcome → auto Hebbian learning
    await tracker.record_outcome(
        decision.id,
        title="Migration succeeded",
        content="Achieved zero downtime",
        success=True,
    )
```

### 2. SQLite — lightweight production

```python
from synaptic import SynapticGraph

graph = SynapticGraph.sqlite("knowledge.db")
await graph.backend.connect()

# RuleBasedClassifier + RelationDetector + Ontology included automatically.
# Just add content — kind and relations are auto-classified.
await graph.add("Refund Policy", "Refunds available within 7 days...")  # → kind=RULE (auto)
```

### 3. Full — LLM classification + embedding + relation detection

```python
from synaptic import SynapticGraph
from synaptic.backends.sqlite import SQLiteBackend
from synaptic.extensions.llm_provider import OllamaLLMProvider

graph = SynapticGraph.full(
    SQLiteBackend("knowledge.db"),
    llm=OllamaLLMProvider(model="qwen3:0.6b"),
    embed_api_base="http://localhost:8080/v1",
    embed_model="BAAI/bge-m3",
)
await graph.backend.connect()

# LLM auto-generates: kind classification + tags + search keywords + search scenarios
# Embeddings include search_keywords → improved vector search accuracy
# Semantic relations auto-detected against existing nodes (DEPENDS_ON, LEARNED_FROM, etc.)
node = await graph.add("Payment Outage Postmortem", "PG API timeout caused...")
```

### 4. Custom — manual composition

Instead of factory methods, compose each component directly:

```python
from synaptic import SynapticGraph, OpenAIEmbeddingProvider
from synaptic.backends.sqlite import SQLiteBackend

graph = SynapticGraph(
    SQLiteBackend("knowledge.db"),
    embedder=OpenAIEmbeddingProvider("http://gpu-server:8080/v1", model="BAAI/bge-m3"),
)
await graph.backend.connect()

# Auto: title + content → vector generation → stored
await graph.add("Deployment Strategy", "Blue-green deployment for zero downtime")
# Auto: query → vector generation → FTS + vector hybrid search
result = await graph.search("deployment approach")
```

### 5. Kuzu — Embedded Property Graph

```python
from synaptic import SynapticGraph

graph = SynapticGraph.kuzu("knowledge.kuzu")
await graph.backend.connect()
await graph.add("Deploy Policy", "Auto-deploy after PR merge")
```

Kuzu runs in-process (like SQLite for graphs) — native openCypher, FTS
and vector indexes via bundled extensions, no server required. MIT licensed.

### 6. Scale — CompositeBackend

```python
from synaptic import SynapticGraph
from synaptic.backends.composite import CompositeBackend
from synaptic.backends.kuzu import KuzuBackend
from synaptic.backends.qdrant import QdrantBackend
from synaptic.backends.minio_store import MinIOBackend

composite = CompositeBackend(
    graph=KuzuBackend("knowledge.kuzu"),
    vector=QdrantBackend("http://localhost:6333"),
    blob=MinIOBackend("localhost:9000", access_key="minio", secret_key="secret"),
)
await composite.connect()

graph = SynapticGraph.full(composite, embed_api_base="http://gpu-server:8080/v1")

# Internal routing:
# - embedding → Qdrant, content > 100KB → MinIO, everything else → Kuzu
```

---

## Architecture

```
SynapticGraph (Facade)
  │
  ├── Auto-Ontology ───── RuleBasedClassifier / LLMClassifier
  │                       RuleBasedRelationDetector / LLMRelationDetector
  ├── OntologyRegistry ── Type hierarchy + property inheritance + constraint validation
  ├── ActivityTracker ─── Session / tool call / decision / outcome capture
  ├── AgentSearch ──────── 6 intent-based search strategies
  ├── HybridSearch ─────── FTS + vector → synonym → LLM rewrite
  ├── ResonanceScorer ──── 5-axis resonance (relevance × importance × recency × vitality × context)
  ├── HebbianEngine ────── Co-activation reinforcement / weakening
  ├── ConsolidationCascade  L0→L3 lifecycle
  ├── EmbeddingProvider ── Auto vector generation (Ollama/vLLM/OpenAI)
  ├── LLMProvider ──────── LLM for ontology construction (Ollama/OpenAI)
  └── Exporters ─────────── Markdown, JSON
       │
  StorageBackend (Protocol)
       │
  ┌────┼──────────┬───────────────┬──────────────┐
  │    │          │               │              │
Memory SQLite  PostgreSQL      Kuzu       CompositeBackend
(dev)  (FTS5)  (pgvector)   (embedded    (Kuzu+Qdrant+MinIO)
                             Cypher)
```

---

## 5-axis Resonance Scoring

```
Score = 0.55 × relevance     Search match score [0,1]
      + 0.15 × importance    (success - failure) / access_count [0,1]
      + 0.20 × recency       exp(-0.05 × days_since_update) [0,1]
      + 0.10 × vitality      Periodic decay ×0.95 [0,1]
      + (context weight) × context  Session tag Jaccard similarity [0,1]
```

Weights vary by intent. `past_failures` emphasizes importance; `context_explore` emphasizes context. **Same query, different intent, different results.**

---

## Ontology

```python
from synaptic import OntologyRegistry, TypeDef, PropertyDef, build_agent_ontology

ontology = build_agent_ontology()

# Add custom type
ontology.register_type(TypeDef(
    name="incident",
    parent="agent_activity",
    description="Production incident",
    properties=[
        PropertyDef(name="severity", value_type="str", required=True),
    ],
))

graph = SynapticGraph(backend, ontology=ontology)
# → Auto-validated on graph.add() and graph.link()
```

### Default Ontology

```
knowledge                          agent_activity
  ├── concept                        ├── session
  ├── entity                         ├── tool_call
  ├── lesson                         ├── observation
  ├── decision                       ├── reasoning
  ├── rule                           └── outcome
  └── artifact
```

## Backends

| Backend | Graph Traversal | Vector Search | Scale | Use Case |
|---------|----------------|--------------|-------|----------|
| `MemoryBackend` | Python BFS | cosine | ~10K | Testing |
| `SQLiteBackend` | CTE recursive | - | ~100K | Embedded (no graph) |
| `KuzuBackend` | Cypher (embedded) | HNSW (optional) | ~10M | **Embedded graph (recommended)** |
| `PostgreSQLBackend` | CTE recursive | pgvector HNSW | ~1M | Production (single DB stack) |
| `QdrantBackend` | - | HNSW + quantization | ~10B | Vector-only |
| `MinIOBackend` | - | - | ~10TB | Blob (S3-compatible) |
| `CompositeBackend` | Kuzu | Qdrant | Unlimited | **Unified router** |

## MCP Server — 16 Tools

```bash
synaptic-mcp                                              # stdio (Claude Code)
synaptic-mcp --db ./knowledge.db                         # SQLite
synaptic-mcp --embed-url http://localhost:8080/v1        # + auto-embedding
```

**Knowledge** (7) — `knowledge_search`, `knowledge_add`, `knowledge_link`, `knowledge_reinforce`, `knowledge_stats`, `knowledge_export`, `knowledge_consolidate`

**Agent Workflow** (4) — `agent_start_session`, `agent_log_action`, `agent_record_decision`, `agent_record_outcome`

**Semantic Search** (3) — `agent_find_similar`, `agent_get_reasoning_chain`, `agent_explore_context`

**Ontology** (2) — `ontology_define_type`, `ontology_query_schema`

## Dev

```bash
uv sync --extra dev --extra sqlite --extra neo4j --extra qdrant --extra minio
uv run pytest -v                              # 266+ tests
uv run pytest tests/benchmark/ -v -s          # Benchmarks (8 datasets + ablation)
uv run ruff check --fix && uv run ruff format
```

## License

MIT
