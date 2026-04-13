# Synaptic Memory — 실전 튜토리얼

30분 안에 Synaptic Memory로 **자신만의 지식 그래프**를 만들고, LLM 에이전트가
그 그래프를 탐색하며 답변하는 것까지 따라할 수 있는 가이드입니다.

---

## 0. 준비

### 필수
```bash
# Python 3.12+
python3 --version

# uv (권장) — https://github.com/astral-sh/uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 설치
```bash
# 최소 구성 (SQLite + 한국어 형태소 분석 + 벡터 인덱스)
uv pip install "synaptic-memory[sqlite,korean,vector,embedding]"

# 또는 전부
uv pip install "synaptic-memory[all]"
```

### (선택) Ollama로 임베딩 모델 띄우기
```bash
# https://ollama.com
ollama pull qwen3-embedding:4b    # 1.5GB
# 서버 주소: http://localhost:11434
```

Ollama 없이도 튜토리얼 진행 가능합니다. 벡터 검색만 빠집니다.

---

## 1. 첫 번째 그래프 — CSV 1개

### 1-1. 데이터 준비

`products.csv` 만들기:

```csv
product_code,name,category,price,description
P001,iPhone 15 Pro,스마트폰,1600000,프리미엄 플래그십 스마트폰
P002,Galaxy Book,노트북,1200000,삼성 노트북
P003,Shin Ramyun,라면,2500,매운 한국 라면
P004,Dried Beef,육류,15000,건조 소고기
P005,CLA Mask,화장품,30000,페이셜 마스크팩
```

### 1-2. 그래프 빌드

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

출력 예시:
```
{'total_nodes': 5, 'kind_entity': 5}
```

5개 상품이 ENTITY 노드로 인제스트됐습니다. 기본 저장 위치는 `synaptic.db`
(SQLite 파일).

### 1-3. 검색해 보기

```python
# tutorial_01_search.py
import asyncio
from synaptic import SynapticGraph
from synaptic.backends.sqlite_graph import SqliteGraphBackend
from synaptic.extensions.evidence_search import EvidenceSearch

async def main():
    backend = SqliteGraphBackend("synaptic.db")
    await backend.connect()
    searcher = EvidenceSearch(backend=backend)

    for query in ["스마트폰", "라면", "매운 음식"]:
        print(f"\n[쿼리] {query}")
        result = await searcher.search(query, k=3)
        for i, ev in enumerate(result.evidence[:3], 1):
            print(f"  {i}. {ev.node.title}  ({ev.score:.3f})")

asyncio.run(main())
```

출력:
```
[쿼리] 스마트폰
  1. products:P001  (0.847)

[쿼리] 라면
  1. products:P003  (0.912)

[쿼리] 매운 음식
  (결과 없음 — 키워드 매칭만으론 못 찾음)
```

"매운 음식"은 안 나옵니다. 벡터 검색을 추가해 봅시다.

### 1-4. 임베딩 붙이기

Ollama가 실행 중이라면:

```python
from synaptic.extensions.embedder import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider(
    api_base="http://localhost:11434/v1",
    model="qwen3-embedding:4b",
)
searcher = EvidenceSearch(backend=backend, embedder=embedder)
```

이제 "매운 음식" → Shin Ramyun을 찾아냅니다. 의미 기반 검색이 활성화됐죠.

---

## 2. 두 번째 그래프 — 다중 테이블 (FK 포함)

### 2-1. 데이터 준비

세 개의 CSV:

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
R001,P001,5,대박 좋음
R002,P001,4,무난
R003,P003,5,매일 먹어요
```

### 2-2. 디렉터리로 한 번에

```python
# tutorial_02.py
import asyncio
from synaptic import SynapticGraph

async def main():
    graph = await SynapticGraph.from_data(
        "./data/",        # 폴더 전체
        db="store.db",
    )
    stats = await graph.stats()
    print(f"Total: {stats['total_nodes']} nodes")

asyncio.run(main())
```

자동으로 3개 테이블 모두 인제스트. 단, **CSV는 FK를 자동 감지하지 못합니다**.
`products.csv`가 먼저 오는지, `product_id` 컬럼이 있는지만 봅니다.

### 2-3. SQL DB로 하면 FK까지 자동

실제 프로덕션 환경이라면 DB에서 바로 인제스트하는 게 편합니다:

```python
graph = await SynapticGraph.from_database(
    "sqlite:///path/to/store.db",
    # 또는 PostgreSQL
    # "postgresql://user:pass@host:5432/dbname"
)
```

PostgreSQL의 `information_schema`를 읽어서 **FK 관계까지 자동 감지**하고
RELATED 엣지를 만듭니다.

### 2-3b. 라이브 DB는 CDC 모드로 (변경분만 동기화)

위 호출은 매번 모든 행을 다시 읽습니다. 한 번 만들고 끝나는 데모/분석에는
괜찮지만, 매 시간 / 매 분 동기화해야 하는 라이브 DB라면 CDC 모드로 바꾸는
게 정석입니다.

```python
# 첫 번째 호출 — deterministic 노드 ID로 풀로드 + sync state 시드
graph = await SynapticGraph.from_database(
    "postgresql://user:pass@host:5432/dbname",
    db="knowledge.db",      # 그래프 SQLite 파일
    mode="cdc",
)

# N번째 호출 — 변경된 행만 다시 읽기
result = await graph.sync_from_database(
    "postgresql://user:pass@host:5432/dbname"
)
print(f"+{result.added} ~{result.updated} -{result.deleted}  ({result.elapsed_ms:.0f}ms)")

for table_stats in result.tables:
    print(f"  {table_stats.table}: strategy={table_stats.strategy}"
          f"  +{table_stats.added} ~{table_stats.updated} -{table_stats.deleted}")
```

동작 원리:

1. **첫 호출 (`mode="cdc"`)**: 모든 행을 읽지만 노드 ID를
   `deterministic_row_id(source_url, table, primary_key)`로 만듭니다.
   같은 행은 다음 호출에서도 같은 ID를 얻으므로 upsert로 동작합니다.
   동시에 그래프 SQLite 안의 `syn_cdc_state` / `syn_cdc_pk_index`
   테이블에 워터마크와 PK 인덱스를 기록합니다.

2. **두 번째 호출부터 (`sync_from_database`)**:
   - `updated_at` 같은 컬럼이 있으면 **timestamp 전략** —
     `WHERE updated_at >= last_watermark` 로 변경분만 읽습니다.
   - 없으면 **hash 전략** — 모든 행을 읽되 row content hash가
     이전과 같은 행은 ingest를 건너뜁니다.
   - 두 전략 모두 **삭제 감지**가 동일하게 동작합니다 (TEMP TABLE
     LEFT JOIN으로 missing PK 찾기).
   - **FK가 바뀐 행**은 옛 RELATED 엣지를 삭제하고 새 엣지를 만듭니다.

3. **`mode="auto"`**: 그래프 파일에 prior CDC 상태가 있으면
   `mode="cdc"`처럼, 없으면 `mode="full"`처럼 동작합니다. 배포
   파이프라인에서 "처음이면 풀로드, 아니면 증분"을 분기 없이
   처리하기 좋습니다.

```python
# 한 줄로 처리 (배포 파이프라인 예시)
graph = await SynapticGraph.from_database(dsn, db="kb.db", mode="auto")
result = await graph.sync_from_database(dsn)
```

#### 검증된 성능 (X2BEE 프로덕션 PostgreSQL, 19,843행)

| | Time |
|---|---|
| Initial CDC load | 51초 |
| Full reload baseline | 35초 |
| **Idempotent re-sync (변경 없음)** | **6초** |
| Search top-1 일치 (vs `mode="full"`) | 4/4 ✓ |

`mode="cdc"`와 `mode="full"`이 동일한 검색 결과를 반환한다는 사실은
`tests/test_cdc_search_regression.py`가 매 PR마다 잠그고 있습니다.

#### 주의: PRIMARY KEY 없는 테이블

소스 스키마에 진짜 PRIMARY KEY가 없는 테이블 (AWS DMS 검증 테이블, 임시
로그 테이블 등) 은 CDC 모드에서 명시적으로 skip됩니다. PK 없이는 행을
안전하게 추적할 수 없기 때문입니다 (`columns[0]`로 fallback하면 unique가
아닐 수 있고 → 같은 노드 ID로 collapse → 행 손실 + 매 동기화마다 churn).

skip된 테이블은 `result.tables`에 `error="no primary key in source schema"`
항목으로 들어갑니다. 검색에 필요한 테이블이라면 ALTER TABLE로 PK를
추가하세요.

### 2-4. 그래프 기반 조인

정형 데이터 도구를 직접 호출해 봅시다:

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

    # 1) iPhone 상품의 판매 이력
    print("\n[iPhone 판매 이력]")
    r = await join_related_tool(
        backend, session,
        from_value="P001",
        fk_property="product_id",
        target_table="sales",
    )
    data = r.to_dict()["data"]
    print(f"  총 {data['total']}건, 표시 {data['showing']}건")
    for item in data["results"]:
        print(f"    {item['title']}: {item['preview'][:60]}")

    # 2) 상품별 판매량 합계
    print("\n[상품별 판매량]")
    r = await aggregate_nodes_tool(
        backend, session,
        table="sales",
        group_by="product_id",
        metric="sum",
        metric_property="quantity",
    )
    data = r.to_dict()["data"]
    for g in data["groups"][:5]:
        print(f"    {g['group']}: {g['value']}개")

    # 3) 5점 리뷰만 상품별로 카운트 (WHERE + GROUP BY)
    print("\n[5점 리뷰 최다 상품]")
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
        print(f"    {g['group']}: {g['value']}건  → {g.get('node_title','')}")

asyncio.run(main())
```

출력:
```
[iPhone 판매 이력]
  총 2건, 표시 2건
    sales:S001: sales: S001 | P001 | 2 | 2024-11-01
    sales:S002: sales: S002 | P001 | 1 | 2024-11-05

[상품별 판매량]
    P003: 5.0개
    P001: 3.0개
    P002: 1.0개

[5점 리뷰 최다 상품]
    P001: 1건  → products:P001
    P003: 1건  → products:P001
```

`aggregate_nodes`의 `node_title` 필드는 FK 해석 결과입니다. 에이전트가 이
ID로 다음 쿼리를 이어갈 수 있습니다.

---

## 3. LLM 에이전트 붙이기

### 3-1. 에이전트란?

여러 도구를 순서대로 호출하면서 답을 찾는 LLM 루프입니다. Synaptic Memory는
도구를 제공하고, LLM이 "어떤 도구를 언제 쓸지"를 판단합니다.

### 3-2. OpenAI 키 준비

```bash
export OPENAI_API_KEY="sk-..."
```

### 3-3. 간단한 에이전트 루프

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
    await agent_query("가장 많이 팔린 상품의 이름은?")
    await agent_query("5점 리뷰가 달린 상품은?")
    await agent_query("iPhone에 관한 리뷰 보여줘")


asyncio.run(main())
```

```bash
uv run python tutorial_03_agent.py
```

에이전트가 도구를 골라가며 답하는 걸 볼 수 있습니다:
```
🙋 가장 많이 팔린 상품의 이름은?
  🔧 T1: aggregate_nodes({'table': 'sales', 'group_by': 'product_id', 'metric': 'sum', 'metric_property': 'quantity'})
  🔧 T2: join_related({'from_value': 'P003', 'fk_property': 'product_id', 'target_table': 'products'})
🤖 가장 많이 팔린 상품은 **Shin Ramyun**입니다 (5개 판매).
```

---

## 4. MCP 서버로 Claude에 붙이기

### 4-1. 서버 실행

```bash
synaptic-mcp --db store.db
# 또는 임베딩 포함
synaptic-mcp --db store.db --embed-url http://localhost:11434/v1
```

### 4-2. Claude Desktop 설정

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

Claude Desktop 재시작. 이제 Claude가 29개 도구를 호출할 수 있습니다.

### 4-3. Claude Code 설정

```bash
claude mcp add synaptic -- synaptic-mcp --db store.db
```

---

## 5. 문서 데이터도 함께

문서를 인제스트하는 3가지 경로가 있습니다:

### 5-1. JSONL로 직접 작성 (의존성 0)

```python
# documents.jsonl — 각 줄이 하나의 문서
# {"title": "...", "content": "...", "category": "..."}

graph = await SynapticGraph.from_data("documents.jsonl")
```

### 5-2. PDF/DOCX/PPTX 파일 직접 (선택 패키지)

```bash
pip install synaptic-memory[docs]   # xgen-doc2chunk 설치
```

```python
graph = await SynapticGraph.from_data("manual.pdf")
graph = await SynapticGraph.from_data("./contracts/")   # 폴더 안의 모든 .pdf/.docx/...
```

지원 형식: PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HWP, HWPX, MD, TXT, RTF.
xgen-doc2chunk가 chunking + 표 보존을 자동 처리합니다.

### 5-3. 자체 파서가 만든 청크 직접 전달 (의존성 0)

LangChain text splitter, Unstructured, 자체 OCR 등을 이미 쓰고 있다면
청크 dict 리스트를 그대로 넘길 수 있습니다:

```python
# 어떤 파서든 (LangChain, Unstructured, 자체 코드)
chunks = my_parser.split("manual.pdf")  # → list[dict]

# 각 dict는 최소 'content' 필드만 있으면 됨.
# 선택: title, doc_id, category, source, chunk_index, page
graph = await SynapticGraph.from_chunks(chunks)
```

위 3가지 모두 자동으로:
- 카테고리 CONCEPT 노드 생성
- 청크 노드 생성 + NFC 정규화
- CONTAINS/PART_OF/NEXT_CHUNK 엣지 구축

검색은 동일한 `deep_search`로:
```python
result = await graph.search("인권경영 기본계획")
# → 관련 문서의 청크가 순서대로 반환
```

### 정형+비정형 혼합

한 디렉터리에 CSV와 JSONL을 섞어 두면 **하나의 그래프**에 들어갑니다.
에이전트가 `filter_nodes`와 `deep_search`를 상황별로 골라 씁니다.

`build_graph_context()`가 자동으로 "이 그래프는 mixed야"라고 알려주기
때문에, 에이전트가 도구를 잘못 쓰는 일이 줄어듭니다.

---

## 6. 품질 튜닝

### 6-1. 임베딩 추가

```python
from synaptic.extensions.embedder import OpenAIEmbeddingProvider

embedder = OpenAIEmbeddingProvider(
    api_base="http://localhost:11434/v1",
    model="qwen3-embedding:4b",
)

graph = await SynapticGraph.from_data(
    "./data/",
    embedder=embedder,  # 인제스트 때 자동 임베딩
)
```

의미 기반 검색이 활성화되어 한국어↔영어 패러프레이즈도 처리됩니다.

### 6-2. Cross-encoder Reranker

TEI 서버가 있다면:

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

KRRA Hard 벤치마크 기준 MRR 0.933 → 1.000 개선 효과가 있었습니다.

### 6-3. 한국어 형태소 분석

기본으로 Kiwi가 사용됩니다 (한글 비율 50% 이상 자동 감지). 설치만 해
두면 됩니다:

```bash
uv pip install "synaptic-memory[korean]"
```

### 6-4. DomainProfile 튜닝

자주 나오는 단어를 stopword로 빼거나, 카테고리를 NodeKind에 매핑:

```toml
# my_domain.toml
name = "my_shop"
locale = "ko"
stopwords_extra = ["상품", "제품", "rows"]

[ontology_hints]
"신상품" = "ENTITY"
"이벤트" = "CONCEPT"
```

```python
from synaptic.extensions.domain_profile import DomainProfile

profile = DomainProfile.load("my_domain.toml")
# 수동 인제스트 시 전달
```

자동 생성도 가능:
```python
from synaptic.extensions.profile_generator import ProfileGenerator

gen = ProfileGenerator()
profile = await gen.generate(name="my_shop", samples=first_20_rows)
```

---

## 7. 평가

자체 데이터로 벤치마크를 돌리려면:

### 7-1. GT(정답) 쿼리 파일 작성

`eval/data/queries/my_queries.json`:
```json
{
  "dataset": "my_dataset",
  "description": "쇼핑몰 검색 테스트",
  "id_field": "node_title",
  "queries": [
    {
      "qid": "q001",
      "query": "가장 많이 팔린 상품",
      "type": "aggregation",
      "relevant_docs": ["products:P003"]
    },
    {
      "qid": "q002",
      "query": "매운 라면",
      "type": "paraphrase",
      "relevant_docs": ["products:P003"]
    }
  ]
}
```

### 7-2. run_all.py에 등록

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

### 7-3. 실행

```bash
uv run python eval/run_all.py --custom-only --embed-url http://localhost:11434/v1

# 에이전트 벤치마크도 같이
uv run python eval/run_all.py --custom-only --agent --judge \
    --openai-key "$OPENAI_API_KEY" \
    --embed-url http://localhost:11434/v1
```

결과 표:
```
Dataset       Corpus  MRR    Hit     Status
My Dataset    2       1.000  2/2     ✅
```

### 7-4. GT 엑셀로 내보내기

```bash
uv run python eval/scripts/export_gt_to_excel.py
# → eval/data/gt_datasets.xlsx
```

각 쿼리 옆에 실제 정답 내용(제목+content 미리보기)이 함께 표시됩니다.

---

## 8. 자주 하는 실수들

### 8-1. "검색해도 결과가 안 나와요"
- Kiwi 설치 확인 (`uv pip install "synaptic-memory[korean]"`)
- 임베딩을 안 썼다면 의미 검색이 없음 → 키워드를 정확히 넣기
- 한국어 데이터인데 영어 쿼리? → 번역해 보기

### 8-2. "그래프가 비어 있어요"
- `stats()` 출력 확인
- `.sqlite-wal` 파일이 큼 → 정상 (SQLite WAL)
- 데이터 파일 경로 확인

### 8-3. "에이전트가 엉뚱한 도구를 써요"
- `build_graph_context()` 출력 확인 → 메타데이터가 제대로 들어가는지
- 시스템 프롬프트에 `filter vs search` 구분이 명확한지
- GPT-4o-mini는 불안정 → GPT-4o나 Claude를 써 보기

### 8-4. "벡터 검색이 너무 느려요"
- usearch 설치 확인 (`uv pip install "synaptic-memory[vector]"`)
- `SqliteGraphBackend._search_vector_hnsw` 경로가 쓰이는지 로그 확인

### 8-5. "M:N 조인이 이상해요"
- DbIngester는 2+ FK 테이블을 자동 감지 → RELATED 엣지로 바로 연결
- CSV의 경우 수동으로 처리 필요 (현재)

---

## 9. 다음 단계

- **멀티턴 에이전트 고도화**: [examples/multi_turn_openai.py](../examples/multi_turn_openai.py)
  실제 벤치마크에 쓰인 완전한 에이전트 코드.
- **29개 도구 전체 탐색**: [../src/synaptic/agent_tools.py](../src/synaptic/agent_tools.py)
  와 [agent_tools_structured.py](../src/synaptic/agent_tools_structured.py).
- **자체 백엔드 만들기**: `StorageBackend` 프로토콜만 구현하면 됩니다.
  [src/synaptic/protocols.py](../src/synaptic/protocols.py) 참고.
- **DomainProfile 작성**: [src/synaptic/extensions/domain_profile.py](../src/synaptic/extensions/domain_profile.py)
- **MCP 서버 커스터마이즈**: [src/synaptic/mcp/server.py](../src/synaptic/mcp/server.py)

---

## 10. 도움이 더 필요하다면

- **GUIDE.md** — 전체 그림이 헷갈리면
- **CONCEPTS.md** — 왜 이렇게 동작하는지 궁금하면
- **ARCHITECTURE.md** — 초기 설계 (Hebbian/Consolidation)
- **GitHub Issues** — 버그 리포트 / 질문
- **CHANGELOG.md** — 버전별 변경 이력
