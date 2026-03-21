# Synaptic Memory v0.5 — 10TB+ Scale Architecture

## 현재 상태

v0.4.0 — 3개 backend (Memory, SQLite, PostgreSQL), Protocol 기반 확장 구조 완성.
PostgreSQL backend에 pgvector + pg_trgm까지 있으나, **단일 서버 한계**.

## 문제

10TB+ 데이터에서 현재 구조가 깨지는 지점:

| 한계 | 현재 | 10TB+ 요구 |
|------|------|-----------|
| Graph 순회 | PostgreSQL CTE (2-hop JOIN) | 수십억 노드에서 multi-hop → O(n²) |
| 벡터 검색 | pgvector HNSW (단일 PG) | 10억+ 벡터 → 단일 서버 메모리 초과 |
| 문서 저장 | Node.content (TEXT 컬럼) | PDF/이미지/코드 → BLOB 비효율 |
| API 스키마 | 미지원 | OpenAPI spec + 응답 캐시 |
| 동시성 | asyncpg pool (단일 서버) | 다중 서비스 동시 읽기/쓰기 |

## 설계 원칙

1. **StorageBackend Protocol 유지** — 기존 코드 변경 0, backend만 추가
2. **용도별 저장소 분리** — 하나의 DB가 모든 걸 하지 않는다
3. **기존 backend 호환** — SQLite/PostgreSQL은 소규모에서 계속 사용
4. **점진적 도입** — 한번에 전부 바꾸지 않고 backend 단위로 추가

## 아키텍처

```
┌──────────────────────────────────────────────────────────────┐
│                   SynapticGraph (Facade)                      │
│   add() · search() · reinforce() · consolidate()             │
│   ← 변경 없음. backend만 교체하면 10TB 지원                    │
└──────────┬───────────────────────────────────────────────────┘
           │
   StorageBackend (Protocol)
           │
   ┌───────┼────────────┬──────────────┬──────────────┐
   │       │            │              │              │
┌──▼──┐ ┌──▼─────┐ ┌───▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│Mem  │ │SQLite  │ │Postgres  │ │Composite │ │Neo4j       │
│(dev)│ │(single)│ │(pgvector)│ │Backend   │ │Backend     │
└─────┘ └────────┘ └──────────┘ │(신규)    │ │(신규)      │
                                └────┬─────┘ └────────────┘
                                     │
                    ┌────────────────┬┴────────────────┐
                    │                │                  │
              ┌─────▼─────┐  ┌─────▼──────┐  ┌───────▼───────┐
              │ Neo4j      │  │ Qdrant     │  │ MinIO         │
              │ (그래프)   │  │ (벡터)     │  │ (문서/blob)   │
              │ Cypher     │  │ HNSW+양자화│  │ S3 호환       │
              └────────────┘  └────────────┘  └───────────────┘
```

## 핵심: CompositeBackend

모든 걸 하나의 DB에 넣지 않는다. **CompositeBackend**가 용도별로 라우팅:

```python
class CompositeBackend:
    """용도별 저장소를 하나의 StorageBackend로 통합."""

    def __init__(
        self,
        graph: Neo4jBackend,      # 노드/엣지 CRUD + 그래프 순회
        vector: QdrantBackend,    # 벡터 검색
        blob: MinIOBackend,       # 대용량 문서/파일
    ) -> None: ...

    # Node CRUD → Neo4j (메타데이터) + MinIO (content가 큰 경우)
    async def save_node(self, node: Node) -> None:
        if len(node.content) > BLOB_THRESHOLD:  # 예: 100KB
            url = await self._blob.upload(node.id, node.content)
            node.content = f"blob://{url}"
        if node.embedding:
            await self._vector.upsert(node.id, node.embedding, node.title)
        await self._graph.save_node(node)

    # Search → 각 엔진에서 검색 후 merge
    async def search_fts(self, query, limit) -> list[Node]:
        return await self._graph.search_fts(query, limit=limit)

    async def search_vector(self, embedding, limit) -> list[Node]:
        ids = await self._vector.search(embedding, limit=limit)
        return await self._graph.get_nodes_batch(ids)

    # Graph traversal → Neo4j (native Cypher, O(1) hop)
    async def get_neighbors(self, node_id, depth) -> list[...]:
        return await self._graph.get_neighbors(node_id, depth=depth)
```

## 신규 Backend 상세

### 1. Neo4jBackend — 그래프 관계 + 메타데이터

**역할**: 노드/엣지 CRUD, FTS, fuzzy, multi-hop 순회
**왜 Neo4j**: index-free adjacency → hop당 O(1), Cypher 쿼리, 10TB+ 검증

```
src/synaptic/backends/neo4j.py

의존성: neo4j[async] (공식 Python 드라이버)
설치: pip install synaptic-memory[neo4j]
```

**스키마 (Cypher)**:
```cypher
// 노드
CREATE (n:Node {
    id: $id, kind: $kind, title: $title,
    content: $content, tags: $tags,
    level: $level, vitality: $vitality,
    access_count: $ac, success_count: $sc, failure_count: $fc,
    source: $source, created_at: $cat, updated_at: $uat
})

// 엣지
CREATE (a)-[r:EDGE {id: $id, kind: $kind, weight: $weight}]->(b)

// 인덱스
CREATE FULLTEXT INDEX node_fts FOR (n:Node) ON EACH [n.title, n.content]
CREATE INDEX node_kind FOR (n:Node) ON (n.kind)
CREATE INDEX node_level FOR (n:Node) ON (n.level)
```

**StorageBackend 매핑**:

| Protocol 메서드 | Neo4j 구현 |
|----------------|-----------|
| `save_node` | `CREATE (n:Node {...})` |
| `get_node` | `MATCH (n:Node {id: $id}) RETURN n` |
| `search_fts` | `CALL db.index.fulltext.queryNodes('node_fts', $query)` |
| `search_fuzzy` | `WHERE n.title CONTAINS $term OR n.content CONTAINS $term` |
| `search_vector` | 미지원 (Qdrant에 위임) |
| `get_neighbors` | `MATCH (n)-[*1..depth]-(m) WHERE n.id = $id RETURN m, r` |
| `prune_edges` | `MATCH ()-[r:EDGE]->() WHERE r.weight < $threshold DELETE r` |
| `decay_vitality` | `MATCH (n:Node) SET n.vitality = n.vitality * $factor` |

### 2. QdrantBackend — 벡터 검색

**역할**: embedding 저장 + ANN 검색
**왜 Qdrant**: Rust 기반 고성능, 양자화(scalar/product) 지원, 10억+ 벡터, gRPC

```
src/synaptic/backends/qdrant.py

의존성: qdrant-client >=1.12
설치: pip install synaptic-memory[qdrant]
```

**구현**:
```python
class QdrantBackend:
    """벡터 검색 전용. StorageBackend의 search_vector만 구현."""

    def __init__(self, url: str, collection: str, dimension: int = 1536): ...

    async def upsert(self, node_id: str, embedding: list[float], title: str) -> None:
        """벡터 + 메타데이터 저장."""

    async def search(self, embedding: list[float], *, limit: int = 20) -> list[str]:
        """ANN 검색 → node_id 목록 반환."""

    async def delete(self, node_id: str) -> None:
        """벡터 삭제."""
```

**Qdrant 설정**:
```python
collection_config = {
    "vectors": {"size": 1536, "distance": "Cosine"},
    "optimizers": {"indexing_threshold": 20000},
    "quantization": {"scalar": {"type": "int8", "always_ram": True}},
}
```

### 3. MinIOBackend — 대용량 문서/파일

**역할**: PDF, 이미지, 코드, API 응답 등 대용량 blob 저장
**왜 MinIO**: S3 호환, 자체 호스팅, 10TB+ 검증, 버전관리

```
src/synaptic/backends/minio.py

의존성: miniopy-async >=1.21
설치: pip install synaptic-memory[minio]
```

**구현**:
```python
class MinIOBackend:
    """S3 호환 blob 저장소."""

    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str): ...

    async def upload(self, node_id: str, content: str | bytes, content_type: str = "text/plain") -> str:
        """파일 업로드 → object URL 반환."""

    async def download(self, node_id: str) -> bytes:
        """파일 다운로드."""

    async def delete(self, node_id: str) -> None:
        """파일 삭제."""

    async def exists(self, node_id: str) -> bool:
        """존재 여부 확인."""
```

**Content 분리 기준**:
- `len(content) < 100KB` → Neo4j Node.content에 직접 저장
- `len(content) >= 100KB` → MinIO에 업로드, Node.content = `"blob://{bucket}/{node_id}"`

### 4. CompositeBackend — 통합 라우터

```
src/synaptic/backends/composite.py
```

**StorageBackend Protocol 완전 구현** — 내부적으로 Neo4j + Qdrant + MinIO 조합:

```python
class CompositeBackend:
    __slots__ = ("_blob", "_blob_threshold", "_graph", "_vector")

    def __init__(
        self,
        graph: Neo4jBackend,
        vector: QdrantBackend | None = None,
        blob: MinIOBackend | None = None,
        blob_threshold: int = 100_000,  # 100KB
    ) -> None: ...

    async def connect(self) -> None:
        await self._graph.connect()
        if self._vector:
            await self._vector.connect()
        if self._blob:
            await self._blob.connect()

    async def close(self) -> None:
        await self._graph.close()
        if self._vector:
            await self._vector.close()
        if self._blob:
            await self._blob.close()
```

## 파일 구조 (추가분)

```
src/synaptic/
  backends/
    neo4j.py          # 신규: Neo4j 그래프 backend
    qdrant.py         # 신규: Qdrant 벡터 backend
    minio.py          # 신규: MinIO blob backend
    composite.py      # 신규: 통합 라우터

tests/
  test_backend_neo4j.py       # @pytest.mark.integration
  test_backend_qdrant.py      # @pytest.mark.integration
  test_backend_minio.py       # @pytest.mark.integration
  test_backend_composite.py   # unit (mock 조합)
```

## pyproject.toml extras 추가

```toml
[project.optional-dependencies]
neo4j = ["neo4j>=5.25"]
qdrant = ["qdrant-client>=1.12"]
minio = ["miniopy-async>=1.21"]
scale = ["neo4j>=5.25", "qdrant-client>=1.12", "miniopy-async>=1.21"]
all = ["aiosqlite>=0.20", "asyncpg>=0.30", "pgvector>=0.3",
       "neo4j>=5.25", "qdrant-client>=1.12", "miniopy-async>=1.21",
       "mcp[cli]>=1.5"]
```

## 사용 예시

### 소규모 (현재 — 변경 없음)
```python
from synaptic.backends.sqlite import SQLiteBackend
graph = SynapticGraph(SQLiteBackend("knowledge.db"))
```

### 10TB+ 프로덕션
```python
from synaptic.backends.neo4j import Neo4jBackend
from synaptic.backends.qdrant import QdrantBackend
from synaptic.backends.minio import MinIOBackend
from synaptic.backends.composite import CompositeBackend

composite = CompositeBackend(
    graph=Neo4jBackend("bolt://neo4j:7687", auth=("neo4j", "password")),
    vector=QdrantBackend("http://qdrant:6333", collection="knowledge"),
    blob=MinIOBackend("minio:9000", bucket="synaptic", access_key="...", secret_key="..."),
)
await composite.connect()
graph = SynapticGraph(composite)

# API는 완전히 동일
await graph.add("API 설계 문서", large_content, kind=NodeKind.ARTIFACT)
result = await graph.search("API 인증", embedding=vec)
```

## 구현 순서

| Step | 내용 | 예상 |
|------|------|------|
| 1 | Neo4jBackend — CRUD + FTS + multi-hop | 핵심, 먼저 |
| 2 | QdrantBackend — upsert + search + delete | 벡터 검색 |
| 3 | MinIOBackend — upload + download + delete | blob 분리 |
| 4 | CompositeBackend — 라우팅 + 통합 테스트 | 조립 |
| 5 | 마이그레이션 스크립트 — SQLite/PG → Composite | 데이터 이관 |
| 6 | 벤치마크 — 100만 노드 성능 비교 | 검증 |

## 인프라 요구사항

```yaml
# docker-compose.yml (개발용)
services:
  neo4j:
    image: neo4j:5-community
    ports: ["7687:7687", "7474:7474"]
    environment:
      NEO4J_AUTH: neo4j/password
    volumes: ["neo4j_data:/data"]

  qdrant:
    image: qdrant/qdrant:v1.12
    ports: ["6333:6333"]
    volumes: ["qdrant_data:/qdrant/storage"]

  minio:
    image: minio/minio
    ports: ["9000:9000", "9001:9001"]
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes: ["minio_data:/data"]
```

## 핵심 설계 결정

1. **SynapticGraph 코드 변경 0** — backend protocol 덕분에 기존 코드 그대로
2. **CompositeBackend가 라우팅** — 호출자는 어떤 DB가 쓰이는지 모른다
3. **blob 분리는 자동** — content 크기에 따라 CompositeBackend가 판단
4. **벡터는 선택적** — Qdrant 없으면 FTS + fuzzy로만 검색 (기존과 동일)
5. **Neo4j Community 사용** — Enterprise 라이선스 없이 시작, 필요 시 전환
