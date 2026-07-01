"""LLM OpenIE entity/triple extractor (opt-in semantic layer).

This is the v0.30 P0 path: it implements the same ``extract_and_link``
protocol as the phrase and spaCy extractors, but it is intentionally not
wired into default ingestion. Callers must inject it explicitly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections import OrderedDict, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING, Any

from synaptic.extensions.entity_ids import (
    canonical_entity_text,
    deterministic_edge_id,
    deterministic_entity_id,
)
from synaptic.models import (
    ConsolidationLevel,
    Edge,
    EdgeKind,
    MemoryEvent,
    MemoryEventKind,
    Node,
    NodeKind,
)

if TYPE_CHECKING:
    from pathlib import Path

    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.extensions.llm_provider import LLMProvider
    from synaptic.graph import SynapticGraph
    from synaptic.protocols import EntityExtractor, StorageBackend

logger = logging.getLogger("entity-extractor-openie")

_PROMPT_VERSION = "openie-v0.30-p0-1"
_MAX_CONTENT_LEN = 2000
_CONFIDENCE_FLOOR = 0.5
_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z가-힣][A-Za-z0-9가-힣_.:/-]{1,}")
_GENERIC_ENTITY_STOPWORDS = {
    "chapter",
    "chunk",
    "content",
    "doc",
    "document",
    "figure",
    "page",
    "section",
    "table",
    "title",
}

_OPENIE_RELATION_MAP: dict[str, EdgeKind] = {
    "related": EdgeKind.RELATED,
    "is_a": EdgeKind.IS_A,
    "type_of": EdgeKind.IS_A,
    "part_of": EdgeKind.PART_OF,
    "belongs_to": EdgeKind.PART_OF,
    "depends_on": EdgeKind.DEPENDS_ON,
    "requires": EdgeKind.DEPENDS_ON,
    "caused": EdgeKind.CAUSED,
    "causes": EdgeKind.CAUSED,
    "produced": EdgeKind.PRODUCED,
    "produces": EdgeKind.PRODUCED,
    "contradicts": EdgeKind.CONTRADICTS,
    "supersedes": EdgeKind.SUPERSEDES,
    "mentions": EdgeKind.MENTIONS,
}

_SYSTEM_PROMPT = """\
Extract a compact OpenIE graph from the provided passage.

Return JSON only:
{
  "entities": [
    {"canonical": "entity name", "type": "person|org|place|product|concept|other", "aliases": [], "confidence": 0.0}
  ],
  "triples": [
    {"subject": "canonical subject", "predicate": "depends_on|part_of|is_a|related|caused|produced|contradicts|supersedes", "object": "canonical object", "confidence": 0.0}
  ]
}

Rules:
- Use canonical names that appear in, or are directly implied by, the passage.
- Prefer fewer high-confidence triples over many weak triples.
- Do not invent facts.
- Use confidence in [0, 1].
- If unsure about a predicate, use "related".
- JSON only. /no_think"""

_OPENIE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "canonical": {"type": "string"},
                    "type": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["canonical"],
            },
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    "required": ["entities", "triples"],
}


@dataclass(slots=True)
class OpenIEEntity:
    """One canonical entity emitted by OpenIE."""

    canonical: str
    entity_type: str = "entity"
    aliases: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(slots=True)
class OpenIETriple:
    """One entity-to-entity OpenIE relation."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


@dataclass(slots=True)
class OpenIEResult:
    """Parsed OpenIE response."""

    entities: list[OpenIEEntity] = field(default_factory=list)
    triples: list[OpenIETriple] = field(default_factory=list)


@dataclass(slots=True)
class OpenIELinkStats:
    """Summary of one OpenIE post-pass over chunk nodes."""

    chunks_scanned: int = 0
    chunks_selected: int = 0
    entity_nodes_touched: int = 0
    entity_node_ids: list[str] = field(default_factory=list)
    edge_ids: list[str] = field(default_factory=list)
    extraction_failures: int = 0
    gated: bool = False
    gate_reason: str = ""
    elapsed_seconds: float = 0.0


@dataclass(slots=True)
class OpenIESelectionPolicy:
    """Deterministic prefilter for expensive OpenIE calls.

    A direct ``OpenIELinker`` construction keeps ``min_candidate_entities=0``
    for simple PoCs. Path-B ingestion builds this from ``DomainProfile`` so
    large corpora default to the DF-prefiltered route.
    """

    min_candidate_entities: int = 0
    max_candidate_df_ratio: float = 1.0
    sample_rate: float = 1.0
    max_chunks: int = 1_000_000
    seed: int = 42

    def __post_init__(self) -> None:
        self.min_candidate_entities = max(0, int(self.min_candidate_entities))
        self.max_candidate_df_ratio = max(0.0, min(1.0, float(self.max_candidate_df_ratio)))
        self.sample_rate = max(0.0, min(1.0, float(self.sample_rate)))
        self.max_chunks = max(0, int(self.max_chunks))
        self.seed = int(self.seed)

    @classmethod
    def from_profile(cls, profile: DomainProfile) -> OpenIESelectionPolicy:
        """Create the OpenIE selector policy declared by a domain profile."""
        return cls(
            min_candidate_entities=getattr(profile, "openie_min_candidate_entities", 2),
            max_candidate_df_ratio=getattr(profile, "openie_max_candidate_df_ratio", 0.3),
            sample_rate=getattr(profile, "openie_sample_rate", 1.0),
            max_chunks=getattr(profile, "openie_max_chunks", 1_000_000),
        )


class _BackendGraphAdapter:
    """Minimal graph facade for extractors that only need ``.backend``."""

    __slots__ = ("backend",)

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend


class ChainedEntityExtractor:
    """Run multiple entity extractors against the same source node.

    Used by Path-A streaming when callers explicitly opt in to OpenIE:
    the deterministic phrase extractor still runs first, then the
    semantic LLM layer adds its revertible OpenIE artifacts.
    """

    __slots__ = ("_extractors",)

    def __init__(self, *extractors: EntityExtractor) -> None:
        self._extractors = tuple(extractors)

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for extractor in self._extractors:
            ids = await extractor.extract_and_link(graph, node_id, title, content)
            for entity_id in ids:
                if entity_id in seen:
                    continue
                seen.add(entity_id)
                out.append(entity_id)
        return out


class OpenIELinker:
    """Path-B OpenIE post-pass over already-ingested CHUNK nodes."""

    __slots__ = (
        "_entity_hint_patterns",
        "_extractor",
        "_max_concurrency",
        "_selection_policy",
        "_selector",
    )

    def __init__(
        self,
        extractor: EntityExtractor,
        *,
        selector: Callable[[Node], bool] | None = None,
        profile: DomainProfile | None = None,
        selection_policy: OpenIESelectionPolicy | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self._extractor = extractor
        self._selector = selector
        self._selection_policy = selection_policy or (
            OpenIESelectionPolicy.from_profile(profile)
            if profile is not None
            else OpenIESelectionPolicy()
        )
        if max_concurrency is None:
            max_concurrency = (
                int(getattr(profile, "openie_max_concurrency", 1))
                if profile is not None
                else 1
            )
        self._max_concurrency = max(1, int(max_concurrency))
        self._entity_hint_patterns = (
            tuple(getattr(profile, "entity_hint_patterns", ())) if profile is not None else ()
        )

    async def link(
        self,
        backend: StorageBackend,
        *,
        source_limit: int = 1_000_000,
    ) -> OpenIELinkStats:
        """Run OpenIE on selected chunks in deterministic backend order."""
        stats = OpenIELinkStats()
        t0 = time()

        chunks = await backend.list_nodes(kind=NodeKind.CHUNK, limit=source_limit)
        chunks.sort(key=lambda node: node.id)
        stats.chunks_scanned = len(chunks)
        if not chunks:
            stats.gated = True
            stats.gate_reason = "no chunk nodes"
            stats.elapsed_seconds = time() - t0
            return stats

        selected = await self._select_chunks(backend, chunks)
        stats.chunks_selected = len(selected)
        if not selected:
            stats.gated = True
            stats.gate_reason = "selector chose no chunks"
            stats.elapsed_seconds = time() - t0
            return stats

        graph = _BackendGraphAdapter(backend)
        if self._max_concurrency > 1 and self._supports_staged_extraction():
            touched = await self._link_with_staged_concurrency(graph, selected, stats)
        else:
            touched = await self._link_serial(graph, selected, stats)

        stats.entity_nodes_touched = len(touched)
        stats.entity_node_ids = sorted(touched)
        stats.edge_ids = await _collect_openie_edge_ids(backend, stats.entity_node_ids, selected)
        await _record_semantic_extract_event(backend, stats, selected, self._extractor)
        stats.elapsed_seconds = time() - t0
        return stats

    async def _link_serial(
        self,
        graph: _BackendGraphAdapter,
        selected: list[Node],
        stats: OpenIELinkStats,
    ) -> set[str]:
        touched: set[str] = set()
        for chunk in selected:
            try:
                ids = await self._extractor.extract_and_link(
                    graph,  # type: ignore[arg-type]
                    chunk.id,
                    chunk.title,
                    chunk.content,
                )
            except Exception:
                logger.warning("OpenIE post-pass failed for chunk %s", chunk.id, exc_info=True)
                stats.extraction_failures += 1
                continue
            touched.update(ids)
        return touched

    def _supports_staged_extraction(self) -> bool:
        return callable(getattr(self._extractor, "extract_for_linking", None)) and callable(
            getattr(self._extractor, "link_result", None)
        )

    async def _link_with_staged_concurrency(
        self,
        graph: _BackendGraphAdapter,
        selected: list[Node],
        stats: OpenIELinkStats,
    ) -> set[str]:
        """Run LLM extraction concurrently, then apply graph writes in input order."""
        extract_for_linking = self._extractor.extract_for_linking  # type: ignore[attr-defined]
        link_result = self._extractor.link_result  # type: ignore[attr-defined]
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def extract(chunk: Node) -> object:
            async with semaphore:
                try:
                    return await extract_for_linking(chunk.id, chunk.title, chunk.content)
                except Exception as exc:  # link loop below owns failure accounting
                    return exc

        outcomes = await asyncio.gather(*(extract(chunk) for chunk in selected))
        touched: set[str] = set()
        for chunk, outcome in zip(selected, outcomes, strict=True):
            if isinstance(outcome, Exception):
                logger.warning(
                    "OpenIE post-pass failed for chunk %s",
                    chunk.id,
                    exc_info=(type(outcome), outcome, outcome.__traceback__),
                )
                stats.extraction_failures += 1
                continue
            try:
                ids = await link_result(
                    graph,  # type: ignore[arg-type]
                    chunk.id,
                    outcome,
                )
            except Exception:
                logger.warning("OpenIE post-pass failed for chunk %s", chunk.id, exc_info=True)
                stats.extraction_failures += 1
                continue
            touched.update(ids)
        return touched

    async def _select_chunks(self, backend: StorageBackend, chunks: list[Node]) -> list[Node]:
        policy = self._selection_policy
        eligible: list[Node] = []
        candidates_by_node: dict[str, set[str]] = {}
        df: defaultdict[str, int] = defaultdict(int)

        for node in chunks:
            if not (node.title or node.content).strip():
                continue
            if self._selector is not None and not self._selector(node):
                continue
            candidates = await _candidate_entities_for_chunk(
                backend,
                node,
                hint_patterns=self._entity_hint_patterns,
            )
            eligible.append(node)
            candidates_by_node[node.id] = candidates
            for candidate in candidates:
                df[candidate] += 1

        if policy.min_candidate_entities <= 0:
            selected = list(eligible)
        else:
            total = max(1, len(eligible))
            max_df_abs = max(1, int(total * policy.max_candidate_df_ratio))
            selected = []
            for node in eligible:
                retained = [
                    candidate
                    for candidate in candidates_by_node[node.id]
                    if df[candidate] <= max_df_abs
                ]
                if len(retained) >= policy.min_candidate_entities:
                    selected.append(node)

        selected = [node for node in selected if _sample_accept(node.id, policy)]
        return selected[: policy.max_chunks]


class LLMOpenIEExtractor:
    """Opt-in LLM extractor that adds entity hubs and typed triples.

    The default graph path never constructs this class. When injected,
    it keeps output contained: low-confidence items are dropped, entity
    ids share the deterministic hub space, and semantic edges get
    deterministic ``openie_`` ids so they can be purged as one layer.
    """

    __slots__ = (
        "_alias_map",
        "_cache",
        "_cache_hits",
        "_cache_misses",
        "_cache_path",
        "_fail_open",
        "_llm",
        "_max_cache_entries",
        "_max_output_tokens",
        "_max_triples_per_chunk",
        "_model_name",
        "_relation_whitelist",
        "_seed",
    )

    def __init__(
        self,
        llm: LLMProvider,
        *,
        seed: int | None = 42,
        alias_map: dict[str, str] | None = None,
        relation_whitelist: set[str] | frozenset[str] | tuple[str, ...] | None = None,
        max_cache_entries: int = 4096,
        max_output_tokens: int = 1024,
        max_triples_per_chunk: int = 24,
        cache_path: Path | None = None,
        fail_open: bool = True,
    ) -> None:
        self._llm = llm
        self._seed = seed
        self._fail_open = fail_open
        self._model_name = str(getattr(llm, "_model", "") or getattr(llm, "model", "") or "")
        self._alias_map = {
            canonical_entity_text(k).lower(): canonical_entity_text(v)
            for k, v in (alias_map or {}).items()
            if canonical_entity_text(k) and canonical_entity_text(v)
        }
        self._relation_whitelist = (
            {str(r).lower().strip() for r in relation_whitelist if str(r).strip()}
            if relation_whitelist
            else set()
        )
        self._max_cache_entries = max_cache_entries
        self._max_output_tokens = max(1, int(max_output_tokens))
        self._max_triples_per_chunk = max_triples_per_chunk
        self._cache_path = cache_path
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._load_cache()

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        """Extract OpenIE entities/triples and link them into ``graph``.

        Returns the entity hub ids touched by this chunk. Failures are
        fail-open: the deterministic ingest path keeps going and no
        semantic layer is added for this chunk.
        """
        result = await self.extract_for_linking(node_id, title, content)
        return await self.link_result(graph, node_id, result)

    async def extract_for_linking(
        self,
        node_id: str,
        title: str,
        content: str,
    ) -> OpenIEResult:
        """Extract OpenIE JSON for a chunk without mutating the graph."""
        text = _chunk_text(title, content)
        if not text.strip():
            return OpenIEResult()

        try:
            return await self.extract(text, title=title)
        except Exception:
            logger.warning("OpenIE extraction failed for node %s", node_id, exc_info=True)
            if not self._fail_open:
                raise
            return OpenIEResult()

    def has_cached_for_linking(self, title: str, content: str) -> bool:
        """Return whether the chunk linking path can be served from cache."""
        text = _chunk_text(title, content)
        return bool(text.strip()) and self.has_cached(text, title=title)

    async def link_result(
        self,
        graph: SynapticGraph,
        node_id: str,
        result: OpenIEResult,
    ) -> list[str]:
        """Materialize an extracted OpenIE result into the graph."""
        if not result.entities and not result.triples:
            return []

        entity_types: dict[str, str] = {}
        aliases: dict[str, str] = {}
        for ent in result.entities:
            canonical = self._canonical(ent.canonical)
            if not canonical or ent.confidence < _CONFIDENCE_FLOOR:
                continue
            entity_types[canonical.lower()] = ent.entity_type or "entity"
            for alias in ent.aliases:
                alias_norm = canonical_entity_text(alias)
                if alias_norm:
                    aliases[alias_norm.lower()] = canonical

        hub_ids: list[str] = []
        touched: set[str] = set()

        # Materialize entity declarations first.
        for ent in result.entities:
            if ent.confidence < _CONFIDENCE_FLOOR:
                continue
            canonical = aliases.get(canonical_entity_text(ent.canonical).lower()) or self._canonical(
                ent.canonical
            )
            if not canonical:
                continue
            hub_id = await self._ensure_entity(
                graph,
                canonical,
                entity_type=ent.entity_type or entity_types.get(canonical.lower(), "entity"),
            )
            if hub_id not in touched:
                touched.add(hub_id)
                hub_ids.append(hub_id)
            await _save_openie_edge(
                graph.backend,
                Edge(
                    id=_openie_edge_id(node_id, hub_id, EdgeKind.MENTIONS),
                    source_id=node_id,
                    target_id=hub_id,
                    kind=EdgeKind.MENTIONS,
                    weight=0.8,
                    properties=self._edge_properties(
                        source_chunk_id=node_id,
                        confidence=ent.confidence,
                        relation="mentions",
                    ),
                    created_at=time(),
                ),
            )

        # Then materialize triples, ensuring endpoints exist even when
        # the model omitted them from ``entities``.
        for triple in result.triples[: self._max_triples_per_chunk]:
            if triple.confidence < _CONFIDENCE_FLOOR:
                continue
            relation = _normalize_relation(triple.predicate)
            if self._relation_whitelist and relation not in self._relation_whitelist:
                continue
            edge_kind = _OPENIE_RELATION_MAP.get(relation, EdgeKind.RELATED)
            if edge_kind == EdgeKind.MENTIONS:
                edge_kind = EdgeKind.RELATED

            subj = aliases.get(canonical_entity_text(triple.subject).lower()) or self._canonical(
                triple.subject
            )
            obj = aliases.get(canonical_entity_text(triple.object).lower()) or self._canonical(
                triple.object
            )
            if not subj or not obj or subj == obj:
                continue
            subj_id = await self._ensure_entity(
                graph,
                subj,
                entity_type=entity_types.get(subj.lower(), "entity"),
            )
            obj_id = await self._ensure_entity(
                graph,
                obj,
                entity_type=entity_types.get(obj.lower(), "entity"),
            )
            for hub_id in (subj_id, obj_id):
                if hub_id not in touched:
                    touched.add(hub_id)
                    hub_ids.append(hub_id)
                await _save_openie_edge(
                    graph.backend,
                    Edge(
                        id=_openie_edge_id(node_id, hub_id, EdgeKind.MENTIONS),
                        source_id=node_id,
                        target_id=hub_id,
                        kind=EdgeKind.MENTIONS,
                        weight=0.8,
                        properties=self._edge_properties(
                            source_chunk_id=node_id,
                            confidence=triple.confidence,
                            relation="mentions",
                        ),
                        created_at=time(),
                    ),
                )

            await _save_openie_edge(
                graph.backend,
                Edge(
                    id=_openie_edge_id(subj_id, obj_id, f"{edge_kind}:{relation}"),
                    source_id=subj_id,
                    target_id=obj_id,
                    kind=edge_kind,
                    weight=max(_CONFIDENCE_FLOOR, min(1.0, float(triple.confidence))),
                    properties=self._edge_properties(
                        source_chunk_id=node_id,
                        confidence=triple.confidence,
                        relation=relation,
                    ),
                    created_at=time(),
                ),
            )

        return hub_ids

    async def extract(self, text: str, *, title: str = "") -> OpenIEResult:
        """Run the LLM and parse an OpenIE response."""
        key = self._cache_key(title, text)
        raw = self._cache.get(key)
        if raw is not None:
            self._cache_hits += 1
            self._cache.move_to_end(key)
            return self._parse_response(raw)
        else:
            self._cache_misses += 1
            raw = await self._llm.generate(
                system=_SYSTEM_PROMPT,
                user=_build_user_prompt(title, text),
                max_tokens=self._max_output_tokens,
                temperature=0.0,
                seed=self._seed,
                response_schema=_OPENIE_SCHEMA,
            )
            result = self._parse_response(raw)
            self._remember(key, raw)
            return result

    def cache_stats(self) -> dict[str, int]:
        """Return per-run cache usage counters for eval/cost monitoring."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "entries": len(self._cache),
        }

    def has_cached(self, text: str, *, title: str = "") -> bool:
        """Return whether ``extract()`` would be cache-only for this input."""
        return self._cache_key(title, text) in self._cache

    async def _ensure_entity(
        self,
        graph: SynapticGraph,
        canonical: str,
        *,
        entity_type: str = "entity",
    ) -> str:
        canonical = self._canonical(canonical)
        hub_id = deterministic_entity_id(canonical)
        existing = await graph.backend.get_node(hub_id)
        if existing is not None:
            if "_openie" not in (existing.tags or []):
                return hub_id
            tags = list(existing.tags or [])
            type_tag = f"_type:{entity_type}"
            if type_tag not in tags:
                tags.append(type_tag)
            existing.tags = tags
            existing.properties = dict(existing.properties or {})
            existing.properties.setdefault("openie_type", entity_type)
            if not existing.title:
                existing.title = canonical
            existing.updated_at = time()
            await graph.backend.update_node(existing)
            return hub_id

        await graph.backend.save_node(
            Node(
                id=hub_id,
                kind=NodeKind.ENTITY,
                title=canonical,
                content="",
                tags=["_openie", "_openie_entity", f"_type:{entity_type}"],
                level=ConsolidationLevel.L0_RAW,
                properties={"openie_type": entity_type},
            )
        )
        return hub_id

    def _canonical(self, text: str) -> str:
        canonical = canonical_entity_text(text)
        return self._alias_map.get(canonical.lower(), canonical)

    def _edge_properties(
        self,
        *,
        source_chunk_id: str,
        confidence: float,
        relation: str,
    ) -> dict[str, str]:
        now = str(time())
        return {
            "source_event_id": "",
            "source_chunk_id": source_chunk_id,
            "extractor": type(self).__name__,
            "model": self._model_name,
            "prompt_version": _PROMPT_VERSION,
            "confidence": str(max(0.0, min(1.0, float(confidence)))),
            "relation": relation,
            "support_count": "1",
            "last_seen_at": now,
            "is_openie": "true",
        }

    def _cache_key(self, title: str, content: str) -> str:
        h = hashlib.sha256()
        h.update(_PROMPT_VERSION.encode("utf-8"))
        h.update(b"\x00")
        h.update(canonical_entity_text(title).encode("utf-8"))
        h.update(b"\x00")
        h.update(content[:_MAX_CONTENT_LEN].encode("utf-8"))
        return h.hexdigest()[:24]

    def _remember(self, key: str, raw: str) -> None:
        self._cache[key] = raw
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_cache_entries:
            self._cache.popitem(last=False)
        if self._cache_path is not None:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self._cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")

    def _load_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            with self._cache_path.open(encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = item.get("key")
                    raw = item.get("raw")
                    if isinstance(key, str) and isinstance(raw, str):
                        self._cache[key] = raw
                        self._cache.move_to_end(key)
            while len(self._cache) > self._max_cache_entries:
                self._cache.popitem(last=False)
        except OSError:
            logger.warning("failed to load OpenIE cache %s", self._cache_path, exc_info=True)

    @staticmethod
    def _parse_response(raw: str) -> OpenIEResult:
        payload = _loads_json_object(raw)
        entities_raw = payload.get("entities", [])
        triples_raw = payload.get("triples", [])

        entities: list[OpenIEEntity] = []
        if isinstance(entities_raw, list):
            for item in entities_raw:
                if not isinstance(item, dict):
                    continue
                canonical = _first_str(item, "canonical", "name", "text", "title")
                if not canonical:
                    continue
                aliases_raw = item.get("aliases", [])
                aliases = [str(a) for a in aliases_raw if isinstance(a, str)]
                entities.append(
                    OpenIEEntity(
                        canonical=canonical,
                        entity_type=_first_str(item, "type", "entity_type", default="entity"),
                        aliases=aliases,
                        confidence=_coerce_confidence(item.get("confidence", 1.0)),
                    )
                )

        triples: list[OpenIETriple] = []
        if isinstance(triples_raw, list):
            for item in triples_raw:
                if not isinstance(item, dict):
                    continue
                subj = _first_str(item, "subject", "subj", "source")
                obj = _first_str(item, "object", "obj", "target")
                pred = _first_str(item, "predicate", "relation", "edge", default="related")
                if not subj or not obj:
                    continue
                triples.append(
                    OpenIETriple(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=_coerce_confidence(item.get("confidence", 1.0)),
                    )
                )

        return OpenIEResult(entities=entities, triples=triples)


async def purge_openie_artifacts(backend: StorageBackend, *, node_limit: int = 1_000_000) -> int:
    """Delete OpenIE semantic edges and OpenIE-created nodes.

    Returns the number of artifacts deleted. Edges are identified by the
    deterministic ``openie_`` id prefix; nodes are identified by the
    ``_openie`` tag. Existing phrase/spaCy hubs touched by OpenIE are
    kept, but their OpenIE relation edges are removed.
    """
    deleted = 0
    nodes = await backend.list_nodes(limit=node_limit)
    edge_ids: set[str] = set()
    node_ids: list[str] = []
    for node in nodes:
        if "_openie" in (node.tags or []):
            node_ids.append(node.id)
        for edge in await backend.get_edges(node.id, direction="both"):
            if edge.id.startswith("openie_"):
                edge_ids.add(edge.id)
    for edge_id in edge_ids:
        await backend.delete_edge(edge_id)
        deleted += 1
    for node_id in node_ids:
        await backend.delete_node(node_id)
        deleted += 1
    return deleted


async def _candidate_entities_for_chunk(
    backend: StorageBackend,
    node: Node,
    *,
    hint_patterns: tuple[re.Pattern[str], ...],
) -> set[str]:
    candidates = _text_candidate_entities(_chunk_text(node.title, node.content), hint_patterns)

    for edge in await backend.get_edges(node.id, direction="outgoing"):
        if edge.kind != EdgeKind.MENTIONS:
            continue
        target = await backend.get_node(edge.target_id)
        if target is None or target.kind != NodeKind.ENTITY:
            continue
        canonical = canonical_entity_text(target.title)
        if canonical:
            candidates.add(canonical.lower())

    return candidates


def _text_candidate_entities(text: str, hint_patterns: tuple[re.Pattern[str], ...]) -> set[str]:
    candidates: set[str] = set()
    for pattern in hint_patterns:
        for match in pattern.finditer(text):
            values = [match.group(0), *match.groups()]
            for value in values:
                if value and _looks_like_candidate_entity(value):
                    candidates.add(canonical_entity_text(value).lower())

    for match in _ENTITY_TOKEN_RE.finditer(text):
        value = match.group(0)
        if _looks_like_candidate_entity(value):
            candidates.add(canonical_entity_text(value).lower())
    return candidates


def _looks_like_candidate_entity(value: str) -> bool:
    candidate = canonical_entity_text(value.strip("()[]{}<>.,;:'\"`"))
    if len(candidate) < 2:
        return False
    lower = candidate.lower()
    if lower in _GENERIC_ENTITY_STOPWORDS:
        return False
    if any("가" <= ch <= "힣" for ch in candidate):
        return len(candidate) >= 2
    if any(ch.isdigit() for ch in candidate):
        return True
    return candidate[:1].isupper() or (candidate.isupper() and len(candidate) >= 2)


def _sample_accept(node_id: str, policy: OpenIESelectionPolicy) -> bool:
    if policy.sample_rate >= 1.0:
        return True
    if policy.sample_rate <= 0.0:
        return False
    digest = hashlib.sha256(f"{policy.seed}\x00{node_id}".encode()).digest()
    score = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return score < policy.sample_rate


async def _collect_openie_edge_ids(
    backend: StorageBackend,
    entity_node_ids: list[str],
    selected_chunks: list[Node],
) -> list[str]:
    edge_ids: set[str] = set()
    for node_id in [*entity_node_ids, *(node.id for node in selected_chunks)]:
        for edge in await backend.get_edges(node_id, direction="both"):
            if edge.id.startswith("openie_"):
                edge_ids.add(edge.id)
    return sorted(edge_ids)


async def _record_semantic_extract_event(
    backend: StorageBackend,
    stats: OpenIELinkStats,
    selected_chunks: list[Node],
    extractor: object,
) -> None:
    save = getattr(backend, "save_memory_event", None)
    if not callable(save):
        return
    properties = {
        "chunks_scanned": str(stats.chunks_scanned),
        "chunks_selected": str(stats.chunks_selected),
        "extraction_failures": str(stats.extraction_failures),
        "linker": "OpenIELinker",
        **_extractor_event_properties(extractor),
    }
    event = MemoryEvent(
        kind=MemoryEventKind.SEMANTIC_EXTRACT,
        source="openie",
        source_id=type(stats).__name__,
        content_hash=_semantic_event_hash(selected_chunks, stats.edge_ids),
        node_ids=stats.entity_node_ids,
        edge_ids=stats.edge_ids,
        confidence=1.0 if stats.extraction_failures == 0 else 0.5,
        properties=properties,
    )
    await save(event)
    await _stamp_openie_edges_with_event(backend, stats.edge_ids, event.id)


def _extractor_event_properties(extractor: object) -> dict[str, str]:
    props = {"extractor": type(extractor).__name__}
    model = str(getattr(extractor, "_model_name", "") or getattr(extractor, "model", "") or "")
    if model:
        props["model"] = model
    prompt_version = str(
        getattr(extractor, "prompt_version", "")
        or getattr(extractor, "_prompt_version", "")
        or ""
    )
    if not prompt_version and hasattr(extractor, "_max_output_tokens"):
        prompt_version = _PROMPT_VERSION
    if prompt_version:
        props["prompt_version"] = prompt_version
    max_output_tokens = getattr(extractor, "_max_output_tokens", None)
    if max_output_tokens is not None:
        props["max_output_tokens"] = str(max_output_tokens)
    max_triples = getattr(extractor, "_max_triples_per_chunk", None)
    if max_triples is not None:
        props["max_triples_per_chunk"] = str(max_triples)
    return props


def _semantic_event_hash(selected_chunks: list[Node], edge_ids: list[str]) -> str:
    h = hashlib.sha256()
    for chunk in selected_chunks:
        h.update(chunk.id.encode("utf-8"))
        h.update(b"\x00")
    for edge_id in edge_ids:
        h.update(edge_id.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


async def _stamp_openie_edges_with_event(
    backend: StorageBackend,
    edge_ids: list[str],
    event_id: str,
) -> None:
    if not edge_ids:
        return
    wanted = set(edge_ids)
    seen: set[str] = set()
    nodes = await backend.list_nodes(limit=1_000_000)
    for node in nodes:
        for edge in await backend.get_edges(node.id, direction="both"):
            if edge.id not in wanted or edge.id in seen:
                continue
            seen.add(edge.id)
            props = dict(edge.properties or {})
            props["source_event_id"] = event_id
            edge.properties = props
            await backend.update_edge(edge)
        if len(seen) >= len(wanted):
            return


async def _save_openie_edge(backend: StorageBackend, edge: Edge) -> bool:
    """Save an OpenIE edge unless a non-OpenIE edge already owns the slot."""
    for existing in await backend.get_edges(edge.source_id, direction="outgoing"):
        if existing.target_id == edge.target_id and existing.kind == edge.kind:
            if not existing.id.startswith("openie_"):
                return False
            existing.weight = max(existing.weight, edge.weight)
            props = dict(existing.properties or {})
            props.update({k: v for k, v in edge.properties.items() if v != ""})
            props["support_count"] = str(_prop_int(props, "support_count", 1) + 1)
            props["last_seen_at"] = str(time())
            existing.properties = props
            await backend.update_edge(existing)
            return True
            break
    await backend.save_edge(edge)
    return True


def _prop_int(props: dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(props.get(key, default)))
    except (TypeError, ValueError):
        return default


def _chunk_text(title: str, content: str) -> str:
    text = f"{title}\n{content}" if content else title
    return text[:_MAX_CONTENT_LEN]


def _build_user_prompt(title: str, text: str) -> str:
    return f"Title: {title}\n\nPassage:\n{text[:_MAX_CONTENT_LEN]}"


def _normalize_relation(value: str) -> str:
    rel = canonical_entity_text(value).lower().replace("-", "_").replace(" ", "_")
    return rel or "related"


def _openie_edge_id(source_id: str, target_id: str, kind: str | EdgeKind) -> str:
    return deterministic_edge_id("openie", source_id, target_id, kind)


def _loads_json_object(raw: str) -> dict[str, Any]:
    text = _strip_json_fence(raw.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    repaired = _salvage_openie_payload(text)
    if repaired is not None:
        return repaired
    msg = f"OpenIE response is not a JSON object: {text[:160]}"
    raise ValueError(msg)


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _salvage_openie_payload(text: str) -> dict[str, Any] | None:
    entities = _salvage_json_array_objects(text, "entities")
    triples = _salvage_json_array_objects(text, "triples")
    if entities is None and triples is None:
        return None
    return {
        "entities": entities or [],
        "triples": triples or [],
    }


def _salvage_json_array_objects(text: str, key: str) -> list[dict[str, Any]] | None:
    key_pos = text.find(f'"{key}"')
    if key_pos < 0:
        return None
    colon_pos = text.find(":", key_pos)
    if colon_pos < 0:
        return None
    array_pos = text.find("[", colon_pos)
    if array_pos < 0:
        return None

    out: list[dict[str, Any]] = []
    pos = array_pos + 1
    while pos < len(text):
        char = text[pos]
        if char == "]":
            return out
        if char != "{":
            pos += 1
            continue
        end = _balanced_json_object_end(text, pos)
        if end < 0:
            return out
        try:
            parsed = json.loads(text[pos : end + 1])
        except json.JSONDecodeError:
            pos = end + 1
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
        pos = end + 1
    return out


def _balanced_json_object_end(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _first_str(item: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return canonical_entity_text(value)
    return default


def _coerce_confidence(value: object) -> float:
    try:
        if isinstance(value, str | int | float):
            return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        pass
    return 1.0
