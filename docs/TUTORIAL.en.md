# Synaptic Memory — Hands-on Tutorial

A 30-minute walkthrough that takes you from zero to your own knowledge graph,
then to an LLM agent that queries it — all with reproducible code.

> Looking for the Korean version? [docs/TUTORIAL.md](TUTORIAL.md).

---

## 0. Prerequisites

### Required
```bash
# Python 3.12+
python3 --version

# uv (recommended) — https://github.com/astral-sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install
```bash
# Minimum useful combo (SQLite + Korean morphology + vector index)
uv pip install "synaptic-memory[sqlite,korean,vector,embedding]"

# Or everything
uv pip install "synaptic-memory[all]"
```

### (Optional) Start Ollama for embeddings
```bash
# https://ollama.com
ollama pull qwen3-embedding:4b    # 1.5 GB
# Server URL: http://localhost:11434
```

You can do the whole tutorial without Ollama — you just won't get vector
search. Everything else (FTS, graph expansion, structured queries) still
works.

---

## 1. Your first graph — a single CSV

### 1-1. Prepare data

Create `products.csv`:

```csv
product_code,name,category,price,description
P001,iPhone 15 Pro,Phone,1600000,Premium flagship smartphone
P002,Galaxy Book,Laptop,1200000,Samsung laptop
P003,Shin Ramyun,Food,2500,Spicy Korean instant ramen
P004,Dried Beef,Meat,15000,Air dried beef jerky
P005,CLA Mask,Cosmetics,30000,Facial sheet mask
```

### 1-2. Build the graph

```python
# tutorial_01.py
import asyncio
from synaptic import SynapticGraph

async def main():
    graph = await SynapticGraph.from_data("products.csv")
    print(await graph.stats())

asyncio.run(main())
```

```bash
uv run python tutorial_01.py
```

Expected output:
```
{'total_nodes': 5, 'kind_entity': 5}
```

Five products landed as ENTITY nodes. The default SQLite file is
`synaptic.db`.

### 1-3. Try a search

```python
# tutorial_01_search.py
import asyncio
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.evidence_search import EvidenceSearch

async def main():
    backend = SqliteGraphBackend("synaptic.db")
    await backend.connect()
    searcher = EvidenceSearch(backend=backend)

    for query in ["smartphone", "ramen", "spicy food"]:
        print(f"\n[query] {query}")
        result = await searcher.search(query, k=3)
        for i, ev in enumerate(result.evidence[:3], 1):
            print(f"  {i}. {ev.node.title}  ({ev.score:.3f})")

asyncio.run(main())
```

Expected output:
```
[query] smartphone
  1. products:P001  (0.847)

[query] ramen
  1. products:P003  (0.912)

[query] spicy food
  (no results — keyword matching alone can't find this)
```

"Spicy food" misses because FTS doesn't know "ramen" and "spicy food"
are semantically related. Let's add embeddings.

### 1-4. Add embeddings

If Ollama is running:

```python
from synaptic.extensions.embedder import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider(
    api_base="http://localhost:11434/v1",
    model="qwen3-embedding:4b",
)
searcher = EvidenceSearch(backend=backend, embedder=embedder)
```

Now "spicy food" returns Shin Ramyun. Semantic search is on.

---

## 2. Second graph — multiple tables with foreign keys

### 2-1. Prepare data

Three CSVs:

**products.csv**
```csv
product_id,name,category
P001,iPhone 15 Pro,Phone
P002,Galaxy Book,Laptop
P003,Shin Ramyun,Food
```

**sales.csv**
```csv
sale_id,product_id,quantity,sold_at
S001,P001,2,2024-11-01
S002,P001,1,2024-11-05
S003,P002,1,2024-11-10
S004,P003,5,2024-11-02
```

**reviews.csv**
```csv
review_id,product_id,score,comment
R001,P001,5,Amazing
R002,P001,4,Fine
R003,P003,5,Eat it every day
```

### 2-2. Ingest a whole directory

```python
# tutorial_02.py
import asyncio
from synaptic import SynapticGraph

async def main():
    graph = await SynapticGraph.from_data(
        "./data/",        # whole folder
        db="store.db",
    )
    stats = await graph.stats()
    print(f"Total: {stats['total_nodes']} nodes")

asyncio.run(main())
```

All three tables are ingested automatically. Note that **CSVs do not
expose foreign keys to the ingester** — it only sees the column name
`product_id`. For proper FK-aware ingestion, use a real database.

### 2-3. Use a real database — FK edges become automatic

```python
graph = await SynapticGraph.from_database(
    "sqlite:///path/to/store.db",
    # or PostgreSQL
    # "postgresql://user:pass@host:5432/dbname"
)
```

For Postgres, Synaptic reads `information_schema` to **discover FK
relationships automatically** and materializes them as RELATED edges.

### 2-3b. Live database? Use CDC mode

The call above rereads every row each time. That's fine for one-shot
demos, but if you need to resync hourly or by the minute, switch to
CDC:

```python
# First call — full load with deterministic node IDs + seed sync state
graph = await SynapticGraph.from_database(
    "postgresql://user:pass@host:5432/dbname",
    db="knowledge.db",      # the graph SQLite file
    mode="cdc",
)

# Every subsequent call — only read what changed
result = await graph.sync_from_database(
    "postgresql://user:pass@host:5432/dbname"
)
print(f"+{result.added} ~{result.updated} -{result.deleted}  ({result.elapsed_ms:.0f}ms)")

for table_stats in result.tables:
    print(f"  {table_stats.table}: strategy={table_stats.strategy}"
          f"  +{table_stats.added} ~{table_stats.updated} -{table_stats.deleted}")
```

How it works:

1. **First call (`mode="cdc"`)**: every row is read, but node IDs are
   generated as `deterministic_row_id(source_url, table, primary_key)`.
   The same source row keeps the same node ID on later calls, so the
   next sync is an upsert. The graph SQLite file also gets
   `syn_cdc_state` / `syn_cdc_pk_index` tables with watermarks and a
   PK index.

2. **Subsequent calls (`sync_from_database`)**:
   - If there is an `updated_at`-style column, the **timestamp
     strategy** fires — `WHERE updated_at >= last_watermark` returns
     only changed rows.
   - Otherwise the **hash strategy** is used — every row is read but
     rows whose content hash matches the previous run are skipped on
     ingest.
   - Both strategies **detect deletes** the same way (TEMP TABLE LEFT
     JOIN to find missing primary keys).
   - **FK changes** tear down the old RELATED edge and create a new
     one.

3. **`mode="auto"`**: if the graph file already has CDC state it
   behaves like `mode="cdc"`, otherwise like `mode="full"`. Handy in
   deployment pipelines that want "first run → full, otherwise →
   incremental" without branching.

```python
# One-liner for pipelines
graph = await SynapticGraph.from_database(dsn, db="kb.db", mode="auto")
result = await graph.sync_from_database(dsn)
```

#### Measured CDC performance (X2BEE production Postgres, 19,843 rows)

| | Time |
|---|---|
| Initial CDC load | 51 s |
| Full reload baseline | 35 s |
| **Idempotent re-sync (no changes)** | **6 s** |
| Top-1 match vs `mode="full"` | 4/4 ✓ |

The regression test `tests/test_cdc_search_regression.py` locks in
that `mode="cdc"` and `mode="full"` return identical top-k on every
PR.

#### Heads up: tables without a PRIMARY KEY

If your source schema has no real primary key on a table (AWS DMS
validation tables, transient log tables, ...), CDC mode **skips it
on purpose**. Without a PK there's no safe way to track a row — a
`columns[0]` fallback could collapse unique-less rows onto the same
node ID, losing data and thrashing on every sync.

Skipped tables show up in `result.tables` with
`error="no primary key in source schema"`. If you need that table in
search, `ALTER TABLE` to add a PK.

### 2-4. Graph-aware joins

Call the structured tools directly:

```python
# tutorial_02_query.py
import asyncio
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.search_session import SearchSession
from synaptic.agent_tools_structured import (
    filter_nodes_tool,
    aggregate_nodes_tool,
    join_related_tool,
)

async def main():
    backend = SqliteGraphBackend("store.db")
    await backend.connect()
    session = SearchSession()

    # 1) Sales history for iPhone
    print("\n[iPhone sales history]")
    r = await join_related_tool(
        backend, session,
        from_value="P001",
        fk_property="product_id",
        target_table="sales",
    )
    data = r.to_dict()["data"]
    print(f"  total {data['total']}, showing {data['showing']}")
    for item in data["results"]:
        print(f"    {item['title']}: {item['preview'][:60]}")

    # 2) Sum of sales per product
    print("\n[sales per product]")
    r = await aggregate_nodes_tool(
        backend, session,
        table="sales",
        group_by="product_id",
        metric="sum",
        metric_property="quantity",
    )
    data = r.to_dict()["data"]
    for g in data["groups"][:5]:
        print(f"    {g['group']}: {g['value']}")

    # 3) Count of 5-star reviews per product (WHERE + GROUP BY)
    print("\n[products with most 5-star reviews]")
    r = await aggregate_nodes_tool(
        backend, session,
        table="reviews",
        group_by="product_id",
        metric="count",
        where_property="score",
        where_op="==",
        where_value="5",
    )
    data = r.to_dict()["data"]
    for g in data["groups"][:5]:
        print(f"    {g['group']}: {g['value']}  → {g.get('node_title','')}")

asyncio.run(main())
```

Output:
```
[iPhone sales history]
  total 2, showing 2
    sales:S001: sales: S001 | P001 | 2 | 2024-11-01
    sales:S002: sales: S002 | P001 | 1 | 2024-11-05

[sales per product]
    P003: 5.0
    P001: 3.0
    P002: 1.0

[products with most 5-star reviews]
    P001: 1  → products:P001
    P003: 1  → products:P001
```

The `node_title` field on `aggregate_nodes` is the resolved FK target
— the agent can chain this ID into the next query.

---

## 3. Attach an LLM agent

### 3-1. What is an agent?

A loop where the LLM decides which tool to call next. Synaptic
provides the tools; the LLM decides *when*.

### 3-2. Set your OpenAI key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3-3. A minimal agent loop

```python
# tutorial_03_agent.py
import asyncio
import json
import os
from openai import AsyncOpenAI

from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.search_session import SearchSession, build_graph_context
from synaptic.agent_tools import search_tool
from synaptic.agent_tools_v2 import deep_search_tool
from synaptic.agent_tools_structured import (
    filter_nodes_tool,
    aggregate_nodes_tool,
    join_related_tool,
)

SYSTEM = """You are a research agent. Use the provided tools to answer.

## Tool selection
- Text question → deep_search
- Price/date/attribute filter → filter_nodes
- "how many per X" / TOP N → aggregate_nodes
- FK-related records → join_related

## Rules
- Use exact table/column names from the metadata below
- Max 5 tool calls
- Reply in the question's language
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "deep_search",
        "description": "Search + expand + read in one call.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "filter_nodes",
        "description": "Filter by property. Returns {total, showing, results}.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"},
            "property": {"type": "string"},
            "op": {"type": "string"},
            "value": {"type": "string"},
        }, "required": ["property", "op", "value"]},
    }},
    {"type": "function", "function": {
        "name": "aggregate_nodes",
        "description": "GROUP BY + COUNT/SUM. Optional WHERE pre-filter.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"},
            "group_by": {"type": "string"},
            "metric": {"type": "string"},
            "where_property": {"type": "string"},
            "where_op": {"type": "string"},
            "where_value": {"type": "string"},
        }, "required": ["group_by"]},
    }},
    {"type": "function", "function": {
        "name": "join_related",
        "description": "FK lookup.",
        "parameters": {"type": "object", "properties": {
            "from_value": {"type": "string"},
            "fk_property": {"type": "string"},
            "target_table": {"type": "string"},
        }, "required": ["from_value", "fk_property", "target_table"]},
    }},
]


async def dispatch(name, args, backend, session):
    if name == "deep_search":
        return await deep_search_tool(backend, session, args["query"])
    if name == "filter_nodes":
        return await filter_nodes_tool(
            backend, session,
            table=args.get("table", ""),
            property=args["property"],
            op=args["op"],
            value=args["value"],
        )
    if name == "aggregate_nodes":
        return await aggregate_nodes_tool(
            backend, session,
            table=args.get("table", ""),
            group_by=args["group_by"],
            metric=args.get("metric", "count"),
            where_property=args.get("where_property", ""),
            where_op=args.get("where_op", ""),
            where_value=args.get("where_value", ""),
        )
    if name == "join_related":
        return await join_related_tool(
            backend, session,
            from_value=args["from_value"],
            fk_property=args["fk_property"],
            target_table=args["target_table"],
        )
    return None


async def agent_query(user_question: str):
    client = AsyncOpenAI()
    backend = SqliteGraphBackend("store.db")
    await backend.connect()
    session = SearchSession(budget_tool_calls=15)

    graph_ctx = await build_graph_context(backend)
    system = SYSTEM + "\n\n" + graph_ctx

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_question},
    ]

    print(f"\n🙋 {user_question}\n")

    for turn in range(5):
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            max_tokens=2048,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"  🔧 T{turn + 1}: {fn}({args})")
                r = await dispatch(fn, args, backend, session)
                if r is None:
                    continue
                result = r.to_dict()
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)[:4000],
                })
        else:
            print(f"\n🤖 {msg.content}\n")
            break


async def main():
    await agent_query("What is the best-selling product?")
    await agent_query("Which products have 5-star reviews?")
    await agent_query("Show me reviews of the iPhone")


asyncio.run(main())
```

```bash
uv run python tutorial_03_agent.py
```

You'll see the agent pick tools in sequence:
```
🙋 What is the best-selling product?
  🔧 T1: aggregate_nodes({'table': 'sales', 'group_by': 'product_id', 'metric': 'sum', 'metric_property': 'quantity'})
  🔧 T2: join_related({'from_value': 'P003', 'fk_property': 'product_id', 'target_table': 'products'})
🤖 The best-selling product is **Shin Ramyun** (5 units sold).
```

---

## 4. MCP server — plug into Claude

### 4-1. Start the server

```bash
synaptic-mcp --db store.db
# or with embeddings
synaptic-mcp --db store.db --embed-url http://localhost:11434/v1
```

### 4-2. Claude Desktop configuration

`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "synaptic": {
      "command": "synaptic-mcp",
      "args": ["--db", "/path/to/store.db", "--embed-url", "http://localhost:11434/v1"]
    }
  }
}
```

Restart Claude Desktop. It now has 36 tools — not just search, but
`knowledge_add_document` / `knowledge_ingest_path` for **adding new
files mid-conversation**, `knowledge_sync_from_database` for **CDC
increments**, and `knowledge_backfill` to **repair embeddings /
phrase hubs** on an existing graph without re-ingesting.

A copy-pasteable config with all three patterns (base, embedding, CDC)
is at [`examples/mcp_claude_desktop.json`](../examples/mcp_claude_desktop.json).

### 4-3. Claude Code configuration

```bash
claude mcp add synaptic -- synaptic-mcp --db store.db
```

---

## 5. Ingesting documents (not just tables)

Three ways to get text into the graph:

### 5-1. JSONL you write yourself (zero dependencies)

```python
# documents.jsonl — one document per line
# {"title": "...", "content": "...", "category": "..."}

graph = await SynapticGraph.from_data("documents.jsonl")
```

### 5-2. Native PDF / DOCX / PPTX / XLSX / HWP (optional extra)

```bash
pip install synaptic-memory[docs]   # installs xgen-doc2chunk
```

```python
graph = await SynapticGraph.from_data("manual.pdf")
graph = await SynapticGraph.from_data("./contracts/")   # every .pdf/.docx/... in the folder
```

Supported formats: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HWP, HWPX,
MD, TXT, RTF. `xgen-doc2chunk` handles chunking and table
preservation automatically.

### 5-3. Bring-your-own chunks (zero dependencies)

If you already use LangChain text splitters, Unstructured, custom OCR,
etc. — just pass the chunk dicts directly:

```python
# any parser (LangChain, Unstructured, your own)
chunks = my_parser.split("manual.pdf")  # → list[dict]

# each dict needs at least a 'content' field.
# Optional: title, doc_id, category, source, chunk_index, page
graph = await SynapticGraph.from_chunks(chunks)
```

All three paths automatically:
- Create category CONCEPT nodes
- Create chunk nodes with NFC normalisation
- Build CONTAINS / PART_OF / NEXT_CHUNK edges

Search is the same:
```python
result = await graph.search("human-rights basic plan")
# → relevant chunks in order
```

### Mixing structured and unstructured

If you drop both CSVs and JSONL into the same directory, they land in
**one graph**. The agent picks `filter_nodes` or `deep_search` per
question. `build_graph_context()` automatically tells the agent "this
is a mixed graph," which cuts down on wrong tool picks.

---

## 6. Quality tuning

### 6-1. Add embeddings

```python
from synaptic.extensions.embedder import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider(
    api_base="http://localhost:11434/v1",
    model="qwen3-embedding:4b",
)

graph = await SynapticGraph.from_data(
    "./data/",
    embedder=embedder,  # auto-embed during ingest
)
```

Semantic search turns on; Korean ↔ English paraphrases start
resolving.

### 6-2. Cross-encoder reranker

If you run a TEI server:

```python
from synaptic.extensions.reranker_cross import TEIReranker
from synaptic.extensions.evidence_search import EvidenceSearch

reranker = TEIReranker(base_url="http://localhost:8080")
searcher = EvidenceSearch(
    backend=backend,
    embedder=embedder,
    reranker=reranker,
)
```

On the KRRA Hard benchmark this moved MRR from 0.933 → 1.000.

### 6-3. Korean morphological analysis

Kiwi is used by default whenever the query is ≥ 50 % Hangul. Just
make sure it's installed:

```bash
uv pip install "synaptic-memory[korean]"
```

### 6-4. DomainProfile tuning

Add stopwords, map categories to `NodeKind`:

```toml
# my_domain.toml
name = "my_shop"
locale = "ko"
stopwords_extra = ["product", "item", "rows"]

[ontology_hints]
"new_arrival" = "ENTITY"
"promotion"   = "CONCEPT"
```

```python
from synaptic.extensions.domain_profile import DomainProfile

profile = DomainProfile.load("my_domain.toml")
# pass it to manual ingesters
```

Auto-generation is also available:
```python
from synaptic.extensions.profile_generator import ProfileGenerator

gen = ProfileGenerator()
profile = await gen.generate(name="my_shop", samples=first_20_rows)
```

---

## 7. Evaluation

To benchmark your own corpus:

### 7-1. Write a ground-truth file

`eval/data/queries/my_queries.json`:
```json
{
  "dataset": "my_dataset",
  "description": "shop search test",
  "id_field": "node_title",
  "queries": [
    {
      "qid": "q001",
      "query": "best-selling product",
      "type": "aggregation",
      "relevant_docs": ["products:P003"]
    },
    {
      "qid": "q002",
      "query": "spicy ramen",
      "type": "paraphrase",
      "relevant_docs": ["products:P003"]
    }
  ]
}
```

### 7-2. Register it in run_all.py

```python
# eval/run_all.py
CUSTOM_DATASETS.append(
    DatasetConfig(
        name="My Dataset",
        path=EVAL_DIR / "data" / "store.db",
        query_path=EVAL_DIR / "data" / "queries" / "my_queries.json",
        is_custom=True,
    ),
)
```

### 7-3. Run it

```bash
uv run python eval/run_all.py --custom-only --embed-url http://localhost:11434/v1

# with the agent benchmark too
uv run python eval/run_all.py --custom-only --agent --judge \
    --openai-key "$OPENAI_API_KEY" \
    --embed-url http://localhost:11434/v1
```

Results table:
```
Dataset       Corpus  MRR    Hit     Status
My Dataset    2       1.000  2/2     ✅
```

### 7-4. Export ground truth to Excel

```bash
uv run python eval/scripts/export_gt_to_excel.py
# → eval/data/gt_datasets.xlsx
```

Each query is paired with the actual relevant text (title + preview),
making it easier to audit the GT.

---

## 8. Common mistakes

### 8-1. "Search returns nothing"
- Is Kiwi installed? (`uv pip install "synaptic-memory[korean]"`)
- No embedder means no semantic search — match keywords exactly
- Korean data with English query? Try translating

### 8-2. "My graph is empty"
- Check `stats()` output
- `.sqlite-wal` getting big is normal (SQLite WAL mode)
- Verify the data path

### 8-3. "The agent picks the wrong tool"
- Inspect `build_graph_context()` — is the metadata complete?
- Is the system prompt explicit about filter vs. search?
- GPT-4o-mini can be flaky — try GPT-4o or Claude

### 8-4. "Vector search is slow"
- Is usearch installed? (`uv pip install "synaptic-memory[vector]"`)
- Check logs that `SqliteGraphBackend._search_vector_hnsw` is the hot
  path

### 8-5. "M:N joins look wrong"
- `DbIngester` auto-detects tables with 2+ FKs and builds RELATED
  edges directly
- CSVs currently require manual handling (no FK inspection)

---

## 9. Next steps

- **Real multi-turn agent** : [examples/multi_turn_openai.py](../examples/multi_turn_openai.py)
  — the full agent used in our benchmarks.
- **All 36 tools**: [src/synaptic/agent_tools.py](../src/synaptic/agent_tools.py),
  [src/synaptic/mcp/server.py](../src/synaptic/mcp/server.py), and
  [agent_tools_structured.py](../src/synaptic/agent_tools_structured.py).
- **Bring your own backend**: implement the `StorageBackend` protocol.
  See [src/synaptic/protocols.py](../src/synaptic/protocols.py).
- **Write a DomainProfile**: [src/synaptic/extensions/domain_profile.py](../src/synaptic/extensions/domain_profile.py)
- **Customise the MCP server**: [src/synaptic/mcp/server.py](../src/synaptic/mcp/server.py)

---

## 10. Getting more help

- **GUIDE.md** — the big picture
- **CONCEPTS.md** — why things work the way they do
- **ARCHITECTURE.md** — original design (Hebbian / consolidation)
- **GitHub Issues** — bug reports / questions
- **CHANGELOG.md** — per-version changes
