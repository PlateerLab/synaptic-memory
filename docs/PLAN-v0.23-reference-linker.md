# PLAN v0.23 — ReferenceLinker: 접속어 패턴 시맨틱 엣지 추출기

작성일: 2026-05-16
상태: **측정 완료 — measured negative. 기본 비활성 유지 (§10 참조).**
선행 분석: `PLAN-v0.17-ontology.md` §3.1 (typed semantic relation 부재), `PLAN-v0.18-architecture.md` §Q2

> **요약 (2026-05-16)**: 모듈·테스트·`--reference-linker` 플래그 구현 완료.
> KRRA(규정 corpus)에서 측정 결과 접속어 패턴은 강하게 작동하나(raw match
> 5,965건) 타깃 해소 정밀도가 ~50%에 머물러 §7 게이트(≤10%)를 통과하지 못함.
> 근본 원인은 corpus 에 **깨끗한 엔티티 인벤토리가 없다**는 것 — phrase-hub
> 노드 7만 개가 노이즈. 검색/agent 에 wiring 하지 않음. 상세 §10.

---

## 0. 한 줄 요약

LLM 호출 없이, 한국어 접속어 패턴을 스캔해 `DEPENDS_ON` / `CAUSED` / `SUPERSEDES` /
`CONTRADICTS` 같은 **typed 시맨틱 엣지**를 corpus 그래프에 추가하는 post-pass 모듈.
EntityLinker 와 동일한 opt-in 방식. 기본 인제스트(`from_data()`)는 비용 0 유지.

---

## 1. 문제 정의 — 왜 필요한가

### 1.1 측정된 현실

`EdgeKind` enum 에는 시맨틱 관계 11종(`CAUSED`, `DEPENDS_ON`, `IS_A`,
`LEARNED_FROM`, `RESULTED_IN`, `PRODUCED`, `CONTRADICTS`, `SUPERSEDES`,
`FOLLOWED_BY`, `INVOKED`, `EXTRACTED_FROM`)이 정의돼 있고, `ppr.py`
`_EDGE_TYPE_WEIGHTS` 와 `ontology.py` `RelationConstraint` 도 이들을 전제한다.

그러나 **corpus 인제스트 경로에서 이 엣지를 만드는 코드가 없다**:

- `DocumentIngester` 는 `backend.save_node/save_edge` 로 직접 쓰며 `graph.add()`
  를 우회한다 → `_relation_detector` (graph.py:981) 가 한 번도 호출되지 않는다.
- `RuleBasedRelationDetector` 는 호출되더라도 산출이 사실상 `RELATED` 뿐이다.
  `DEPENDS_ON` 규칙은 `source.kind == RULE && target.kind == CONCEPT` 게이트라
  `CHUNK` 노드(문서 청크 전부)에는 절대 발화하지 않는다.
- `LLMRelationDetector` 는 corpus 경로에 미연결 + 노드당 LLM 호출 비용.

결론: `from_data()` 로 만든 모든 그래프의 엣지는 `PART_OF` / `CONTAINS` /
`NEXT_CHUNK` / `RELATED`(FK) / `MENTIONS`(opt-in) 뿐이다. 이는 **파일시스템
디렉토리 트리 + DB FK 그래프**이지 온톨로지가 아니다.

### 1.2 무엇을 노리고, 무엇을 노리지 않는가

**노리지 않음**: 단일샷 MRR 도약. `PLAN-v0.17-ontology.md` 가 측정한 바,
시맨틱 메커니즘 3종(phrase extractor −6.6%, decomposer −10.6%, entity linker
−4%)이 MuSiQue 에서 전부 음수였다. HippoRAG2 분석상 typed relation 자체의
알파는 ~+5%p 수준. **단일샷 회귀 0 이 성공 기준**이지 개선이 목표가 아니다.

**노림**: agent 모드의 관계 항해(traversal) affordance. 측정상 agent 81% vs
단일샷 40%. agent 가 "이 규정에 의존하는 항목" / "이 결정이 야기한 결과" 를
명시적으로 따라갈 typed 엣지가 현재 0개다. `follow_tool` 은 이미 `edge_kind`
인자를 받으므로, **엣지만 존재하면 agent 가 즉시 활용 가능**하다.

---

## 2. 메커니즘 — 접속어 → EdgeKind

한국어는 관계 접속어가 명시적 형태소다. 영어 대비 rule-based 정밀도가 높다.
`DomainProfile.reference_patterns` (`(.+?)에 따라`, `(.+?)에 의거`)는 이미
존재하지만 **어디서도 소비되지 않는 shelfware** — 그 인프라를 살린다.

### 2.1 내장 접속어 테이블 (locale ko / multi 일 때만 활성)

| EdgeKind | 접속어 큐 | 의미 |
|---|---|---|
| `DEPENDS_ON` | `…에 따라`, `…에 의거(하여)`, `…에 근거(하여)`, `…를 준용` | source 가 참조 규정에 의존 |
| `CAUSED` | `…(으)로 인해`, `… 때문에`, `…의 결과(로)`, `…에 기인` | 참조가 source 를 야기 |
| `SUPERSEDES` | `…를 개정`, `…를 대체(하여)`, `…를 폐지하고`, `…를 갈음` | source 가 참조를 대체 |
| `CONTRADICTS` | `…와 달리`, `…에 반(하여\|해)`, `…와 배치` | source 가 참조와 상충 |

`reference_patterns` (기존 프로필 필드)는 `DEPENDS_ON` 버킷에 추가 주입 —
하위호환. 프로필 TOML 로 버킷별 패턴 확장 가능 (§5.2).

### 2.2 캡처 범위 — greedy 회피

`(.+?)에 따라` 의 `.+?` 는 문장 경계를 넘어 과대 캡처한다. 대신 접속어 **직전의
명사구**만 bounded window 로 잡는다:

```
([가-힣A-Za-z0-9·()「」『』 ]{2,40}?)\s*(?:에 따라|에 의거하여|…)
```

캡처 span 은 후처리로 trim: 선행 조사/공백 제거, 마지막 명사 토큰 경계까지.

---

## 3. 타깃 해소 (Target Resolution) — 가장 어려운 부분

캡처 span("개인정보 보호법", "안전관리규정 제3조")을 **기존 노드 id** 로
해소해야 엣지를 달 수 있다. 정밀도 우선: 확신이 없으면 **엣지를 버린다**.

해소 우선순위 (먼저 맞는 것 채택):

1. **정확 title 매칭** — 후보 노드(아래)의 title 인덱스에 span 이 정확히 존재.
   `relation_detector.InvertedIndex._title_index` 재사용 가능.
2. **Phrase-hub 매칭** — EntityLinker 가 선행 실행됐다면 span 이 `_phrase`
   태그 ENTITY 허브일 수 있다. `_phrase_hub_id(span)` 로 직접 조회.
3. **포함 매칭(bounded)** — span 이 후보 title 을 부분문자열로 포함하거나
   그 역. 단 후보 title 길이 ≥ 6 (짧은 title false positive 차단), 매칭
   후보가 유일할 때만.
4. 위 셋 다 실패 → **드롭** (stats.unresolved++).

**후보 노드 집합** = 해소 대상이 될 수 있는 노드:
- `DOCUMENT` 노드 (= `ontology_hints` 로 `RULE`/`DECISION` kind 가 된 문서 포함)
- 카테고리 노드 (`cat_*`)
- EntityLinker 허브 (`_phrase` 태그) — 있으면

자기 자신(같은 doc 의 청크)으로의 엣지는 제외.

---

## 4. 모듈 설계 — `ReferenceLinker`

`EntityLinker` 와 동일한 형태의 post-pass. 새 파일
`src/synaptic/extensions/reference_linker.py`.

### 4.1 3-pass 알고리즘

```
Pass 1 — 타깃 인덱스 구축
  후보 노드(DOCUMENT/category/phrase-hub) 를 list_nodes 로 모아
  title → node_id, (phrase_hub_id) → node_id 인덱스 생성.

Pass 2 — 소스 스캔 + 패턴 매칭
  source_kind(기본 CHUNK) 노드를 순회.
  각 텍스트에 접속어 패턴 적용 → (span, edge_kind) 후보 추출.
  span 정규화(trim).

Pass 3 — 해소 + typed 엣지 발행
  각 (span, edge_kind) 를 §3 으로 해소.
  해소 성공 시 deterministic id 로 Edge(source→target, kind) 발행.
  소스당 edge_kind 별 cap (기본 5) 적용. save_edges_batch.
```

`MENTIONS` 와 달리 **새 노드를 만들지 않는다** — 기존 노드 사이에만 엣지를 건다.
재실행 멱등성: edge id = `blake2b(f"{kind}:{src}->{tgt}")`.

### 4.2 코드 스케치

```python
# src/synaptic/extensions/reference_linker.py
"""Connective-pattern semantic edge extractor (LLM-free post-pass).

Scans source nodes for Korean discourse connectives and emits typed
edges (DEPENDS_ON / CAUSED / SUPERSEDES / CONTRADICTS) between existing
nodes. Creates no new nodes. Idempotent (deterministic edge ids).
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field

from synaptic.models import Edge, EdgeKind, NodeKind

# --- built-in Korean connective table (locale ko/multi only) -----------
_NP = r"[가-힣A-Za-z0-9·()「」『』 ]{2,40}?"

_DEFAULT_KO_PATTERNS: dict[EdgeKind, list[str]] = {
    EdgeKind.DEPENDS_ON: [
        rf"({_NP})\s*에 (?:따라|의거하여|의거해|근거하여|근거해)",
        rf"({_NP})\s*을?를? 준용",
    ],
    EdgeKind.CAUSED: [
        rf"({_NP})\s*(?:으로|로) 인(?:해|하여)",
        rf"({_NP})\s* 때문에",
        rf"({_NP})\s*의 결과(?:로)?",
    ],
    EdgeKind.SUPERSEDES: [
        rf"({_NP})\s*을?를? (?:개정|대체하여|폐지하고|갈음)",
    ],
    EdgeKind.CONTRADICTS: [
        rf"({_NP})\s*(?:와|과) 달리",
        rf"({_NP})\s*에 반(?:하여|해)",
    ],
}


@dataclass(slots=True)
class ReferenceLinkStats:
    source_nodes_scanned: int = 0
    raw_matches: int = 0
    resolved: int = 0
    unresolved: int = 0
    edges_created: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def _ref_edge_id(kind: EdgeKind, src: str, tgt: str) -> str:
    h = hashlib.blake2b(f"{kind.value}:{src}->{tgt}".encode(), digest_size=8)
    return f"ref_{h.hexdigest()}"


def _normalize_span(span: str) -> str:
    s = span.strip().strip("「」『』()")
    # drop trailing josa fragments left by lazy capture
    return re.sub(r"\s+", " ", s).strip()


class ReferenceLinker:
    __slots__ = ("_compiled", "_max_per_kind_per_source", "_profile")

    def __init__(self, profile, *, max_per_kind_per_source: int = 5) -> None:
        self._profile = profile
        self._max_per_kind_per_source = max_per_kind_per_source
        # built-in ko table + profile.reference_patterns (→ DEPENDS_ON)
        table: dict[EdgeKind, list[str]] = {
            k: list(v) for k, v in _DEFAULT_KO_PATTERNS.items()
        }
        for p in getattr(profile, "reference_patterns", ()):  # existing field
            table[EdgeKind.DEPENDS_ON].append(p.pattern)
        # profile.relation_patterns override hook (§5.2) merged here
        self._compiled: list[tuple[EdgeKind, re.Pattern[str]]] = [
            (kind, re.compile(pat))
            for kind, pats in table.items()
            for pat in pats
        ]

    async def link(
        self, backend, *, source_kind=NodeKind.CHUNK, source_limit=1_000_000
    ) -> ReferenceLinkStats:
        stats = ReferenceLinkStats()
        t0 = time.time()

        # locale gate — connectives only reliable for Korean text
        if self._profile.locale not in ("ko", "multi"):
            stats.elapsed_seconds = time.time() - t0
            return stats

        # --- Pass 1: target title index -------------------------------
        title_index: dict[str, str] = {}        # title_lower -> node_id
        for kind in (NodeKind.ENTITY, NodeKind.RULE, NodeKind.DECISION,
                     NodeKind.CONCEPT, NodeKind.OBSERVATION):
            for n in await backend.list_nodes(kind=kind, limit=source_limit):
                t = n.title.strip().lower()
                if len(t) >= 4:
                    title_index.setdefault(t, n.id)
        # DOCUMENT-kind nodes are typically ENTITY w/ doc_id; handled above.
        # category nodes:
        cats = await backend.list_nodes(kind=NodeKind.CONCEPT, limit=source_limit)

        # --- Pass 2 + 3: scan, match, resolve, emit -------------------
        sources = await backend.list_nodes(kind=source_kind, limit=source_limit)
        stats.source_nodes_scanned = len(sources)
        new_edges: list[Edge] = []
        now = time.time()

        for src in sources:
            text = (src.title + "\n" + src.content) if src.title else src.content
            if not text.strip():
                continue
            per_kind: dict[EdgeKind, int] = {}
            seen_targets: set[str] = set()
            for kind, rx in self._compiled:
                for m in rx.finditer(text):
                    stats.raw_matches += 1
                    span = _normalize_span(m.group(1))
                    tgt = self._resolve(span, title_index, src.id)
                    if tgt is None:
                        stats.unresolved += 1
                        continue
                    if tgt in seen_targets:
                        continue
                    if per_kind.get(kind, 0) >= self._max_per_kind_per_source:
                        continue
                    per_kind[kind] = per_kind.get(kind, 0) + 1
                    seen_targets.add(tgt)
                    stats.resolved += 1
                    stats.by_kind[kind.value] = stats.by_kind.get(kind.value, 0) + 1
                    new_edges.append(Edge(
                        id=_ref_edge_id(kind, src.id, tgt),
                        source_id=src.id, target_id=tgt,
                        kind=kind, weight=0.7, created_at=now,
                    ))

        await backend.save_edges_batch(new_edges)
        stats.edges_created = len(new_edges)
        stats.elapsed_seconds = time.time() - t0
        return stats

    def _resolve(self, span, title_index, src_id):
        if len(span) < 2:
            return None
        s = span.lower()
        # 1. exact title
        if s in title_index and title_index[s] != src_id:
            return title_index[s]
        # 2. phrase-hub id (EntityLinker hubs share the deterministic id)
        from synaptic.extensions.entity_linker import _phrase_hub_id
        hub = _phrase_hub_id(span)
        # caller may verify hub existence via title_index reverse / get_node
        # 3. bounded substring — unique containment, target title >= 6 chars
        hits = [nid for t, nid in title_index.items()
                if len(t) >= 6 and (t in s or s in t) and nid != src_id]
        if len(hits) == 1:
            return hits[0]
        return None
```

> 위는 스케치다. `_resolve` 의 phrase-hub 경로는 hub 존재 확인(`get_node`)이
> 필요하고, 대형 corpus 에서 substring 매칭은 `title_index` 역인덱스(노드 title
> 토큰 → id)로 최적화해야 한다. 구현 시 정리.

---

## 5. 통합 지점

### 5.1 호출 — opt-in, EntityLinker 와 동일 패턴

`eval/run_all.py` 에 `--reference-linker` 플래그 추가, EntityLinker post-pass
바로 다음에서 호출:

```python
if args.reference_linker:
    from synaptic.extensions.reference_linker import ReferenceLinker
    rl = ReferenceLinker(profile)
    stats = await rl.link(backend, source_kind=_NK.CONCEPT)
    logger.info("ReferenceLinker: %d edges (%s)", stats.edges_created, stats.by_kind)
```

기본 `from_data()` 에는 **넣지 않는다** — 비용 0 인덱싱 원칙 유지. 추후
`SynapticGraph.from_data(..., link_references=True)` kwarg 로 노출 검토.

### 5.2 DomainProfile 확장 (선택)

내장 ko 테이블로 충분하면 프로필 변경 불필요. 도메인별 접속어를 추가하려면
TOML 에 새 섹션:

```toml
[relation_patterns]
DEPENDS_ON = ["(.+?)을 적용받아"]
SUPERSEDES = ["(.+?)을 전면 개편"]
```

`domain_profile.py` 에 `relation_patterns: dict[str, tuple[Pattern, ...]]`
필드 추가 + 직렬화. 하위호환 — 없으면 빈 dict.

### 5.3 검색 파이프라인 (별도 게이트, 신중히)

- **PPR**: `_EDGE_TYPE_WEIGHTS` 가 이미 `CAUSED`/`DEPENDS_ON`/`SUPERSEDES`/
  `CONTRADICTS` 가중치를 가지므로 **엣지 생성만으로 PPR 이 자동 반영**한다.
  추가 코드 0. 단일샷 회귀 여부를 여기서 측정.
- **GraphExpander**: `_expand_related` 와 별도로 `_expand_typed` 추가 — typed
  엣지 1-hop 확장, `reason="typed_relation"`. `hybrid_reranker._REASON_PRIOR`
  에 `"typed_relation": 0.5` 추가. **이 단계는 PPR-only 측정에서 단일샷 회귀가
  0 임을 확인한 뒤에만 켠다.**

---

## 6. Agent `follow` 도구 확장

`follow_tool` (`agent_tools.py:736`)은 이미 `edge_kind` 를 받아 임의 EdgeKind
를 항해한다 — typed 엣지가 생기면 **변경 없이 작동**. 단 agent 가 어떤 typed
엣지가 존재하는지 *발견*할 방법이 없다. 작은 보강 2가지:

1. `follow_tool` / `expand` 결과에 `available_edge_kinds` 필드 추가 — 해당
   노드의 outgoing/incoming 엣지 종류 집합을 함께 반환. agent 가 "이 노드에서
   `DEPENDS_ON` 으로 갈 수 있다" 를 알게 된다.
2. agent priming snapshot (v0.18-α2, `618a0dc`)에 그래프의 typed 엣지 분포
   1줄 추가: `"typed edges: DEPENDS_ON×120, CAUSED×34, ..."`.

도구 신설은 불필요 — `follow` 로 충분.

---

## 7. 측정 계획

| 항목 | 측정 | 성공 기준 |
|---|---|---|
| 인제스트 비용 | KRRA/assort 빌드 시간 | EntityLinker 대비 +20% 이내 |
| 엣지 수율 | `stats.by_kind`, resolved/raw 비율 | resolved ≥ 40% (precision 우선) |
| 단일샷 회귀 | `run_all.py --quick --local-bge --reference-linker` 14-bench | 평균 MRR Δ ≥ −0.005 (회귀 0 이 목표) |
| agent 효과 | KRRA Hard / X2BEE Hard agent solved | 관계형 쿼리에서 +pp, 전체 회귀 0 |
| 정밀도 샘플 | resolved 엣지 30건 수동 검수 | 오결합 ≤ 10% |

`PLAN-v0.17` 의 측정 규율 준수: temp=0/seed=42, corpus hash 고정, baseline 대비.

---

## 8. 리스크 & 솔직한 기대값

- **단일샷 도약 없음**: §1.2 참조. 이 작업의 ROI 를 단일샷 MRR 로 평가하면
  실패로 보인다. 평가축은 agent traversal 과 "회귀 0".
- **한국어 한정**: 영어 접속어는 형태소가 불명확 → `locale` 게이트로 ko/multi
  만. 영어 multi-hop(MuSiQue)은 여전히 §Q2 OpenIE 트랙 소관.
- **타깃 해소 정밀도**: greedy 캡처 + 약한 매칭은 오결합을 낳는다. 완화 =
  bounded window + 4단계 해소 게이트 + 미해소 시 드롭. 30건 수동 검수 필수.
- **패턴 유지보수**: 접속어 테이블이 코드에 박힘. 도메인 확장은 §5.2 프로필로.
- **shelfware 가능성**: EntityLinker 가 벤치에서 ±1% 였듯, ReferenceLinker 도
  measured negative 면 release scope 에서 기본 비활성 유지 + 문서에 기록
  (`CONCEPTS.md` §13 measured negatives 관례).

---

## 9. 작업 분해

1. `reference_linker.py` 모듈 + `ReferenceLinkStats` (스케치 → 완성, `_resolve`
   역인덱스 최적화 포함)
2. `tests/test_reference_linker.py` — 패턴 매칭/해소/멱등성/locale 게이트
3. `run_all.py --reference-linker` 플래그 + post-pass 호출
4. 측정 라운드 1 — PPR-only, 14-bench 단일샷 회귀 확인
5. 회귀 0 확인 시: `_expand_typed` + reranker reason prior, `follow`
   `available_edge_kinds` 보강, agent 측정
6. 결과를 `CONCEPTS.md` / 베이스라인에 기록, scope 결정

(1)~(4) 가 1차 PR. (5)~(6) 은 측정 게이트 통과 후 2차.

---

## 10. 측정 결과 — 2026-05-16 (measured negative)

### 10.1 측정 라운드 1 — `run_all.py --quick --reference-linker`

5개 public 데이터셋 전부 **0 edges**. KRRA/assort/X2BEE 는 `run_public_dataset`
경로를 거치지 않아 ReferenceLinker 가 아예 실행되지 않음. public RAG 데이터셋은
passage 가 `doc_id` 로만 제목이 붙어 해소 타깃(엔티티-이름 노드)이 0개.
단일샷 회귀 0 (0 edges = no-op).

### 10.2 측정 라운드 2 — KRRA 그래프 직접 측정

`krra_graph.sqlite` 사본에 직접 실행 (CHUNK 18,600개 스캔).

| 버전 | clean 타깃 | raw match | 해소율 | 엣지 | 정밀도(20 샘플) |
|---|---:|---:|---:|---:|---|
| v1 (lazy capture + substring) | 61,886 | 10,708 | 16.9% | 1,813 | ~50% (불통과) |
| **v2 (window + clean dict filter)** | 67,515 | 5,965 | **91.9%** | 5,484 | **~50% (불통과)** |
| v3 (CONCEPT 카테고리만 타깃) | 155 | 5,965 | 0.2% | 11 | 高 but recall 무용 |

### 10.3 근본 원인 — 진단 확정

접속어 패턴 자체는 **언어학적으로 건전** — 규정 corpus 에서 raw match 5,965건.
문제는 전부 **타깃 해소**다:

- KRRA 의 ENTITY phrase-hub 70,405개가 깨끗한 엔티티가 아니다. `하는 계약`,
  `상이하게 납품된`, `준하여 지급한다`, `직급별로 근무평` 같은 **문법 파편 /
  잘린 단어**가 다수. EntityLinker 의 phrase 추출은 FTS 노이즈 감소용이지
  ontology 인벤토리가 아니다.
- 어미 기반 clean 필터(`_is_clean_target`)는 71,345 → 67,515 로 5%만 제거.
  `하는 계약`(명사 `계약`으로 끝남), `직급별로 근무평`(명사형 음절로 끝남)처럼
  **어미로 안 끝나는 파편**은 거를 수 없다.
- v3 처럼 타깃을 깨끗한 CONCEPT 카테고리(155개)로 좁히면 정밀도는 오르지만
  18,600 청크에서 엣지 11개 — recall 이 무의미.

**중간 granularity 의 깨끗한 엔티티 레이어가 corpus 에 존재하지 않는다.**
연결의 품질은 엔티티 사전 품질에 종속되는데, 그 사전이 없다.

### 10.4 결론 — `PLAN-v0.18` §Q2 와 수렴

이 결과는 `PLAN-v0.17-ontology.md` §3.3 의 HippoRAG2 분석과 정확히 일치한다 —
"진짜 기여는 query→**clean** triple linking 이고, clean triple 을 얻으려면 LLM
OpenIE 가 필요하다." ReferenceLinker 의 접속어-타이핑 절반(어떤 관계인가)은
작동하지만, 나머지 절반(무엇을 가리키는가)은 깨끗한 노드 인벤토리 없이는 불가.

**결정**:
- ReferenceLinker 는 `--reference-linker` opt-in 으로 repo 에 유지 (기본 OFF).
  EntityLinker 가 measured ±1% 임에도 유지되는 것과 동일한 관례.
- §9 의 작업 5~6 (검색/agent wiring) **취소** — 정밀도 게이트 미통과.
- 향후: `PLAN-v0.18` §Q2 OpenIE 트랙이 깨끗한 엔티티/triple 노드를 만들면,
  ReferenceLinker 의 접속어 테이블을 그 위에 재측정할 가치가 있다. 접속어
  타이핑은 LLM-free 이므로 OpenIE triple 의 술어(predicate) 보강에 쓰일 수 있다.
- `CONCEPTS.md` §13 measured negatives 에 1줄 기록.
