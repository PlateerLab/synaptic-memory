# Hybrid Knowledge-Graph Upgrade — Implementation Plan
**synaptic-memory: deterministic structural core + opt-in LLM semantic layer**

Scope note: this folds in every `needs_rework` fix from the review and is grounded in verified source (`llm_provider.py:19`, `graph.py` `full()` final `cls(...)`, `store.py:33` `add_node(..., node_id=None)`, `entity_linker.py:268/280`, `entity_extractor_spacy.py:272`, `document_ingester.py:287`). Where the original design was wrong, it is corrected, not papered over.

---

## 1. Architecture

```
                         ┌─────────────────────────────────────────────────────┐
                         │            DETERMINISTIC STRUCTURAL CORE              │
                         │                (default, zero-LLM, replay-safe)      │
INGEST                   │                                                       │
 ├─ Path A (streaming)   │  CATEGORY ──CONTAINS──▶ DOCUMENT ──CONTAINS──▶ CHUNK  │
 │  SynapticGraph.add()  │      ▲                      │  PART_OF / NEXT_CHUNK   │
 │  graph.py:1325-1343   │      │                      ▼                         │
 │                       │  StructuralReferenceLinker (REFERENCES citations,     │
 ├─ Path B (bulk 10TB)   │     gated by reference_key_property + collision gate) │
 │  DocumentIngester     │  PhraseExtractor / EntityLinker → ENTITY hubs +       │
 │  document_ingester.py │     MENTIONS (DF-statistical, deterministic ids)      │
 │     :447 tail         │                                                       │
 └───────────────────────┴──────────────────────┬────────────────────────────────┘
                                                 │  (structural edges = authority)
        opt-in flag (llm + openie_enabled)       │
        ════════════════════════════════════════ ▼ ════════════════════════════════
                         ┌─────────────────────────────────────────────────────┐
                         │         OPT-IN LLM SEMANTIC LAYER (ADD-ONLY)         │
EXTRACT                  │  LLMOpenIEExtractor (entity_extractor_openie.py)     │
 (selective, sampled)    │   local Qwen via vLLM → {entities[], triples[]}       │
                         │   coref+canonical, temp=0, content-hash cache         │
                         │            │                                          │
MERGE                    │   shares ONE deterministic hub-id space with          │
 (structural precedence) │   EntityLinker/spacy  → no split hubs                 │
                         │   typed ENTITY──IS_A/PART_OF/DEPENDS_ON/RELATED──▶ENTITY│
                         │   confidence floor ≥0.5 → Edge.weight; below = DROP    │
                         │   ALL nodes/edges tagged "_openie" (revertible)        │
                         │   NEVER mutates REFERENCES/CONTAINS/PART_OF structural │
                         └──────────────────────┬──────────────────────────────────┘
                                                ▼
RETRIEVE   ppr.py (edge.weight × EDGE_TYPE_WEIGHTS[kind]) — unchanged map;
           graph_expander.py 1-hop entity mentions + RELATED;
           EvidenceSearch → HybridReranker.  OpenIE edges enter PPR as normal
           edges; RELATED=0.4 noise bucket auto-contains junk.
```

**End-to-end flow:** ingest writes structural core (always) → optional selective LLM-OpenIE pass extracts entity/triples over sampled chunks → merge layer attaches them to the **same** entity hubs the deterministic linker uses, with structural edges winning any conflict → retrieval consumes a single enriched graph through the existing PPR/expander, no retrieval-side rewrite.

---

## 2. Key decisions & the central tradeoff

**What zero-LLM purity is given up — precisely:**
- **Only** when a caller explicitly sets `llm != None` **and** `openie_enabled=True` (two flags). The default `SynapticGraph.full(backend)` and all of Path B without an injected `openie_extractor` stay **byte-for-byte zero-LLM**. This is guaranteed by a regression test (P0 exit criterion), not by assertion.
- The LLM layer is **ADD-ONLY**: it creates `ENTITY` hubs and `MENTIONS`/typed `ENTITY→ENTITY` edges. It **never** creates, mutates, or deletes `REFERENCES`/`CONTAINS`/`PART_OF`/`NEXT_CHUNK`. So finreg's 73% structural-REFERENCES result is mechanically untouchable by this layer.

**What is NOT given up:**
- Bit-identical replay on the default path (no LLM in the loop).
- Structural determinism. On the LLM path, replay-determinism is achieved through `temperature=0 + fixed seed + content-hash cache` — but this requires extending the provider first (see below). Honest caveat: **LLM-path determinism is "deterministic given identical model weights + identical sampler params + cache"**, which is weaker than the structural path's pure-function determinism. We do not claim bit-identical across different vLLM builds.

**Structural-precedence rule (crisp):**
> On any edge conflict between an OpenIE edge and a structural edge for the same `(source, target)`, the structural edge wins. Mechanism: OpenIE always runs **after** `StructuralReferenceLinker` (Path B) and emits edges of a **different `kind`**; sqlite's `ON CONFLICT(source_id, target_id, kind) DO UPDATE` keys on `kind`, so an OpenIE `RELATED` edge can never overwrite a structural `REFERENCES` edge. Additionally, all OpenIE output carries tag `_openie`, making the entire semantic layer one-query revertible.

**The central tradeoff:** we accept a *bounded, contained, revertible* injection of LLM noise (gated to RELATED=0.4 + confidence floor ≥0.5 + structural precedence) in exchange for closing the entity-composition multi-hop gap (MuSiQue R@5 0.453 → target ≥0.6). We do **not** route exact/structural relations through the LLM — those keep using the deterministic graph that already beats the LLM-entity baseline.

---

## 3. Phased roadmap

### P0 — Smallest PoC that proves entity-hop closes with NO structural regression (~1 week)
**What:** A minimal Path-B OpenIE post-pass on a small Korean+English sample, sharing the deterministic hub-id space.

**Files:**
1. `extensions/llm_provider.py:19` — **extend the protocol first** (blocking prerequisite):
   `async generate(*, system, user, max_tokens=1024, temperature=0.0, seed=None, response_schema=None) -> str`. Update Ollama (`options: {temperature, seed}`, `format=schema`), OpenAI/vLLM (`temperature, seed, response_format=json_schema/guided_json`), Anthropic (temperature only). Backward-compatible defaults so existing callers are untouched.
2. `extensions/entity_extractor_openie.py` (new) — `LLMOpenIEExtractor` implementing `EntityExtractor.extract_and_link` (protocols.py:155); `OpenIEEntity/Triple/Result` dataclasses; content-hash LRU + optional on-disk cache (copy `classifier_llm.py:433`, **add `_PROMPT_VERSION` into the key**); `_OPENIE_RELATION_MAP` (mirror `relation_detector_llm.py:26`, unknown→RELATED); try/except→fallback.
3. **Shared deterministic hub id** — add `ent_{md5(type\x00nfc(canonical))[:16]}` helper and have OpenIE pass it as `add_node(..., node_id=...)` (the kwarg **already exists**, `store.py:33`). Retrofit `entity_extractor_spacy.py:272` and `entity_linker.py:268` to mint the same deterministic id (today spacy=uuid, linker=`md5(phrase)` with no type prefix → they split). This is the fix for the split-hub bug.
4. `domain_profile.py` — `openie_enabled: bool = False`, `openie_alias_map`, `openie_relation_whitelist`.

**Why:** Proves the two load-bearing claims at once — entity-hop recall rises **and** the deterministic top-10 is unchanged — on a corpus small enough to run on a 4090.

**Exit criteria:**
- Regression test: ingest sample twice with `llm=None` → identical top-10 (existing bit-identical claim holds post-PR).
- Same corpus with OpenIE on: a hand-built 5-question entity-composition set goes from ≥3 fails → ≤1 fail.
- Structural finreg-style citation set: recall **unchanged** (Δ = 0).
- `DELETE FROM ... WHERE tags LIKE '%_openie%'` fully restores the pre-OpenIE graph (revertibility proven).

### P1 — Production Path-B post-pass + selective extraction (~2 weeks)
**What:** Promote P0 into an `EntityLinker`-sibling post-pass with sampling/DF-prefilter for scale.

**Files:**
- `extensions/document_ingester.py:447` (after `StructuralReferenceLinker.link`) — call optional profile-gated OpenIE post-pass; **add `openie_extractor` to `__slots__`** (`:287`, confirmed needed) + ctor kwarg.
- New `OpenIELinker` (sibling of `EntityLinker`, `entity_linker.py:97` shape): walks `list_nodes(kind=CHUNK)`, applies `selector` (DF-prefilter / sampling / chunk-importance), `extract_many` with `asyncio.Semaphore` bounded concurrency, deterministic input-order reassembly (not completion order — else replay breaks), returns a `gated`-flagged Stats dataclass.
- **Cross-chunk triple buffering:** `extract_and_link` ensures **both** subject and object hubs exist (content-free, idempotent, deterministic id) so triples whose object appears in a later chunk never dangle.

**Why:** 10TB needs selective extraction; Path B after structural linking is the correct home (recon-confirmed).

**Exit:** runs on a 10–50GB slice within GPU budget; structural no-regress holds; MuSiQue-style R@5 up measurably.

### P2 — Path-A streaming wiring + query→entity dense linking (~2 weeks)
**What:** Optional streaming OpenIE + close the remaining MuSiQue gap via query→entity dense linking.

**Files:**
- `graph.py` `full()` — compose OpenIE **with** PhraseExtractor **only** under `if llm is not None and openie_enabled`; default leaves `phrase_extractor=PhraseExtractor()` untouched. **Path-A OpenIE OFF by default even with llm set** (cost guard).
- Entity-hub embeddings (canonical titles) + `query→entity` dense linking in `graph_expander.py` / EvidenceSearch.

**Why:** README states closing MuSiQue needs "OpenIE triples + query→triple dense linking" — P2 adds the second half.

**Exit:** MuSiQue-style R@5 ≥ 0.6; no Path-A unthrottled per-chunk LLM on bulk corpora.

---

## 4. Eval plan

| Track | Dataset | Metric | Gate |
|---|---|---|---|
| **Structural NO-REGRESS** | finreg-style citation multi-hop (the 73% set) | recall/answerable | **Δ ≥ 0** vs current. Any regression = block. |
| **Default determinism** | any corpus, `llm=None`, 2× ingest | top-10 identity | **bit-identical** (hard gate) |
| **Entity-hop UP** | MuSiQue-style entity-composition | R@5 | **0.453 → ≥0.60** (P2); ≥0.52 (P1) |
| **Korean PoC** | 마사회(saleskit) + 법령/규정 sample | answerable@5 on hand-built entity-hop Qs | ≥+30%p over dense-only |
| **Noise containment** | inject known-false triples | structural-set recall delta | **0** (precedence + floor works) |
| **Revertibility** | `DELETE WHERE tag _openie` | graph diff | restores pre-OpenIE exactly |

Dev loop uses a small local model for iteration; final numbers re-run with the deployable on-prem Qwen. Extraction-quality probe on a 도로공사/국보연 sample **before** locking the relation whitelist + confidence floor (open question #4).

---

## 5. Air-gap & 10TB cost model

**Serving:** local Qwen via vLLM, OpenAI-compatible endpoint, `guided_json`/`response_format=json_schema`, `temperature=0`, fixed `seed`. No cloud. Point existing `OpenAILLMProvider`/a vLLM provider at the on-prem endpoint.

**Determinism legs (all three required):** `temperature=0` + fixed `seed` + content-hash cache (`sha256(title+content[:2000]+_PROMPT_VERSION)[:24]`). On-disk cache shared across air-gapped cluster nodes → re-ingest free + reproducible. **Honest caveat:** first-extraction of a cache-miss chunk is only deterministic if the vLLM build/sampler is pinned; cache is what guarantees replay, so the on-disk cache must be treated as a build artifact, not throwaway.

**10TB math (why LLM-on-every-chunk is infeasible, and the fix):**
- 10TB / ~2KB per chunk ≈ **5×10⁹ chunks**. At even 50ms/chunk on one GPU that's ~8 GPU-years. Per-chunk LLM is a non-starter — **selective extraction is mandatory, not optional.**
- Selector policy (P1): DF-prefilter (only chunks containing ≥2 candidate entities from the deterministic NER pass) + importance sampling + dedup via content-hash cache. Target extraction rate **1–5%** of chunks → 5×10⁷–2.5×10⁸ calls.
- Throughput: a single H100 with vLLM batched guided-JSON at ~short outputs ≈ thousands of chunks/min. At 1% extraction + caching, a 10TB corpus is **days–weeks on a small GPU pool**, not years. The cache makes re-runs O(new chunks only).
- Path A (streaming) stays for small/interactive volume only; bulk always goes Path B with the selector.

---

## 6. Risk register

| Risk | Severity | Mitigation (concrete) |
|---|---|---|
| **Graph pollution via split hubs** (review issue #2) | HIGH | Converge OpenIE + spacy + EntityLinker on ONE deterministic hub id `ent_{md5(type,nfc(canonical))}`, passed via the existing `add_node(node_id=...)` kwarg (`store.py`). MENTIONS edges then genuinely collapse; PPR mass concentrates. P0 exit gate verifies. |
| **Determinism loss** (review issue #1) | HIGH | Extend `LLMProvider.generate` with `temperature/seed/response_schema` BEFORE building OpenIE (blocking P0 task). Without it, temp=0/seed/grammar are unreachable and the cache only helps on hits. |
| **Cost blowup on 10TB** (review issue #5) | HIGH | Path-A OpenIE OFF by default even with llm; bulk forced through Path-B selector (DF-prefilter+sampling, 1–5%) + content-hash cache. Unified cost story: A=interactive small, B=bulk throttled. |
| **Hallucinated triples** | MED | Confidence floor **≥0.5 → DROP** (not just down-weight) + unknown predicate→RELATED(0.4) + structural precedence (`ON CONFLICT kind`). Tag `_openie` = one-query purge. |
| **Dangling cross-chunk triples** (review issue #3) | MED | Buffer/ensure both endpoint hubs exist (content-free, idempotent, deterministic id) before emitting entity→entity edge. |
| **Default path regressed by careless wiring** (review issue #4) | MED | `full()` composes OpenIE only under `if llm is not None and openie_enabled`; `llm=None` leaves `phrase_extractor=PhraseExtractor()` byte-for-byte. Regression test in P0. |
| **slots AttributeError** (review issue #6) | LOW | Add `openie_extractor` to `DocumentIngester.__slots__` (`:287`) with the ctor kwarg. |
| **Cache staleness on prompt change** (open Q#1) | LOW | Fold `_PROMPT_VERSION` into cache key; version bump = scoped re-extract. |

**Dropped from the original design as flawed:** the claim that OpenIE/phrase MENTIONS edges auto-collapse "for free" (false until hub-id convergence lands — issue #2); the assumption that the existing provider can pass seed/guided_json (it cannot — issue #1); unthrottled Path-A wiring on bulk (issue #5).

---

## 7. The very first PoC to run this week

**Scope:** Path-B-only, single file + provider extension, no streaming, no `full()` changes.
1. Extend `llm_provider.py` `generate()` with `temperature=0, seed`, `response_schema` (Ollama + one vLLM/OpenAI impl).
2. Write `entity_extractor_openie.py` with the shared deterministic `ent_` hub id, content-hash cache, confidence floor ≥0.5, unknown→RELATED.
3. Retrofit `entity_extractor_spacy.py:272` / `entity_linker.py:268` to mint the same deterministic id via `add_node(node_id=...)`.

**Data:** ~200–500 chunks from the **마사회 saleskit** corpus (already on hand, user_id 86 / 177) **plus** a small 법령/규정 sample (Korean entity-composition is the real target). Run the local Qwen already available on the home server.

**Success numbers (all three must hold):**
- **Determinism:** `llm=None` ingest ×2 → identical top-10. (hard gate)
- **Entity-hop:** a hand-built 8-question entity-composition set (e.g. "X가 만든 상품을 담당하는 부서") improves from baseline by **≥4 questions answerable@5**.
- **No structural regression:** the finreg/citation-style subset recall delta = **0**, and `DELETE WHERE tags LIKE '%_openie%'` reproduces the pre-OpenIE graph exactly.

If determinism or structural-regress fails, stop and fix before any scaling work — those two gates are the whole premise of the hybrid design.

---

**Verified source anchors:** `llm_provider.py:19` (no temp/seed — gap real), `graph.py` `full()` final `cls(phrase_extractor=PhraseExtractor())` unconditional (wiring critique correct), `store.py:33` `add_node(..., node_id=None)` accepts explicit deterministic id → `ON CONFLICT(id) DO UPDATE` (split-hub fix is cheap), `entity_linker.py:268` `_phrase_hub_id=md5(phrase)[:16]` vs `entity_extractor_spacy.py:272` uuid (they split today), `document_ingester.py:287` `__slots__` (slots edit confirmed needed).
