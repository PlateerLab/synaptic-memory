"""Post-processing entity linker — DF-filtered hub creation.

Walks an already-ingested graph, runs a phrase extractor over selected
source nodes (typically ``NodeKind.CHUNK``), computes distinct-source DF
for each candidate phrase, filters by ``profile.min_df`` /
``profile.max_df_ratio``, and materializes surviving phrases as hub
``NodeKind.ENTITY`` nodes with ``EdgeKind.MENTIONS`` edges back to the
sources that contained them.

Use this **post-hoc** mode when:

- Inline extraction (``PhraseExtractor`` on ``graph.add``) would create
  too many noisy single-occurrence phrases.
- You want global DF pruning to suppress boilerplate.
- You ingested a corpus without extraction and want to add entities
  later without re-ingesting.

The linker is locale- and domain-agnostic — the only coupling to Korean
or English is whatever extractor you inject. A ``DomainProfile`` supplies
the DF thresholds and (indirectly via the extractor) the stopword /
metadata strip configuration.

Example::

    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.extensions.entity_linker import EntityLinker
    from synaptic.extensions.phrase_extractor import create_phrase_extractor

    profile = DomainProfile.load("profiles/my_corpus.toml")
    extractor = create_phrase_extractor(profile)
    linker = EntityLinker(extractor=extractor, profile=profile)

    stats = await linker.link(backend, source_kind=NodeKind.CHUNK)
    print(stats.phrase_nodes_created, stats.mentions_edges_created)
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from synaptic.models import ConsolidationLevel, Edge, EdgeKind, Node, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.protocols import StorageBackend


class PhraseExtractorProtocol(Protocol):
    """Minimal contract any injected extractor must satisfy.

    Both :class:`EnglishPhraseExtractor` (via the ``extract`` method added
    in v0.12) and :class:`KoreanPhraseExtractor` implement this.
    """

    def extract(self, text: str) -> set[str]:
        """Return the distinct phrases found in ``text``."""
        ...


@dataclass(slots=True)
class EntityLinkStats:
    """Summary of one ``EntityLinker.link`` run."""

    source_nodes_scanned: int = 0
    raw_phrase_candidates: int = 0
    kept_phrases: int = 0
    phrase_nodes_created: int = 0
    mentions_edges_created: int = 0
    elapsed_seconds: float = 0.0
    top_phrases_by_df: list[tuple[str, int]] = field(default_factory=list)


class EntityLinker:
    """Post-processing pass that builds hub ENTITY nodes via DF filtering.

    Args:
        extractor: A phrase extractor implementing
            ``extract(text) -> set[str]``. The linker does not care
            whether it is English, Korean, or LLM-backed.
        profile: Domain profile supplying ``min_df`` and ``max_df_ratio``
            thresholds. Other profile fields are ignored here (they were
            meant for the extractor).
        max_links_per_source: Upper bound on MENTIONS edges per source
            node (prevents edge explosion on chunk-rich corpora).
    """

    __slots__ = ("_extractor", "_profile", "_max_links_per_source")

    def __init__(
        self,
        *,
        extractor: PhraseExtractorProtocol,
        profile: DomainProfile,
        max_links_per_source: int = 15,
    ) -> None:
        self._extractor = extractor
        self._profile = profile
        self._max_links_per_source = max_links_per_source

    async def link(
        self,
        backend: StorageBackend,
        *,
        source_kind: str | NodeKind = NodeKind.CHUNK,
        source_limit: int = 1_000_000,
    ) -> EntityLinkStats:
        """Walk source nodes, compute DF, filter, materialize hubs.

        Three-pass algorithm:

        1. **Scan** — ``backend.list_nodes(kind=source_kind)`` then
           ``extractor.extract(title + content)`` for each.
        2. **Filter** — keep phrases whose distinct-source count is in
           ``[profile.min_df, profile.max_df_ratio * total]``.
        3. **Materialize** — create one hub ``ENTITY`` node per surviving
           phrase (deterministic id so re-runs are idempotent), then
           emit ``MENTIONS`` edges (source → hub) up to
           ``max_links_per_source`` per source.

        The passes use ``save_nodes_batch`` / ``save_edges_batch`` so
        backends that implement real batch writes can accelerate. For
        backends whose batch is just a loop (``kuzu.py``) the result is
        still correct, just not faster.

        Args:
            backend: Graph backend to read from and write to.
            source_kind: Which node kind to scan. Defaults to ``CHUNK``.
            source_limit: Safety fuse on listing (defaults large).

        Returns:
            ``EntityLinkStats`` describing the run.
        """
        stats = EntityLinkStats()
        t0 = time.time()

        source_nodes = await backend.list_nodes(kind=source_kind, limit=source_limit)
        stats.source_nodes_scanned = len(source_nodes)
        if not source_nodes:
            stats.elapsed_seconds = time.time() - t0
            return stats

        # Pass 1 — extract phrases per source, build DF map
        df_map: defaultdict[str, list[str]] = defaultdict(list)
        for src in source_nodes:
            text = src.title + "\n" + src.content if src.title else src.content
            if not text.strip():
                continue
            phrases = self._extractor.extract(text)
            for phrase in phrases:
                df_map[phrase].append(src.id)

        stats.raw_phrase_candidates = len(df_map)

        # Pass 2 — DF filter
        total = len(source_nodes)
        max_df_abs = max(1, int(total * self._profile.max_df_ratio))
        min_df = self._profile.min_df

        kept: dict[str, list[str]] = {}
        for phrase, sources in df_map.items():
            df = len(sources)
            if min_df <= df <= max_df_abs:
                kept[phrase] = sources

        stats.kept_phrases = len(kept)
        if not kept:
            stats.elapsed_seconds = time.time() - t0
            return stats

        # Top phrases for reporting (by DF descending)
        sorted_by_df = sorted(kept.items(), key=lambda kv: -len(kv[1]))
        stats.top_phrases_by_df = [(p, len(s)) for p, s in sorted_by_df[:30]]

        # Pass 3a — materialize hub phrase nodes (deterministic ids)
        phrase_to_hub_id: dict[str, str] = {}
        new_nodes: list[Node] = []
        now = time.time()
        for phrase, sources in kept.items():
            hub_id = _phrase_hub_id(phrase)
            phrase_to_hub_id[phrase] = hub_id
            new_nodes.append(
                Node(
                    id=hub_id,
                    kind=NodeKind.ENTITY,
                    title=phrase,
                    content="",
                    tags=["_phrase"],
                    level=ConsolidationLevel.L0_RAW,
                    properties={"df": str(len(sources))},
                    created_at=now,
                    updated_at=now,
                )
            )

        await backend.save_nodes_batch(new_nodes)
        stats.phrase_nodes_created = len(new_nodes)

        # Pass 3b — emit MENTIONS edges, capped per source.
        # Invert the map to source → phrases so we can cap per source
        # and prefer longer (more specific) phrases first.
        source_to_phrases: defaultdict[str, list[str]] = defaultdict(list)
        for phrase, sources in kept.items():
            for src_id in sources:
                source_to_phrases[src_id].append(phrase)

        new_edges: list[Edge] = []
        for src_id, phrases in source_to_phrases.items():
            # Prefer longer phrases — more content-bearing, less noise
            phrases.sort(key=lambda p: (-len(p), p))
            for phrase in phrases[: self._max_links_per_source]:
                hub_id = phrase_to_hub_id[phrase]
                new_edges.append(
                    Edge(
                        id=_mention_edge_id(src_id, hub_id),
                        source_id=src_id,
                        target_id=hub_id,
                        kind=EdgeKind.MENTIONS,
                        weight=0.8,
                        created_at=now,
                    )
                )

        await backend.save_edges_batch(new_edges)
        stats.mentions_edges_created = len(new_edges)

        stats.elapsed_seconds = time.time() - t0
        return stats


def _phrase_hub_id(phrase: str) -> str:
    """Deterministic 16-char id for a phrase hub node.

    Stable across runs so re-linking the same corpus is idempotent —
    backends that upsert on primary key will simply overwrite the same
    hub rather than duplicating. The ``phrase:`` prefix makes these ids
    greppable in diagnostic dumps.
    """
    h = hashlib.md5(phrase.encode("utf-8")).hexdigest()[:16]
    return f"phrase_{h}"


def _mention_edge_id(source_id: str, hub_id: str) -> str:
    """Deterministic edge id so repeated ``link()`` runs are idempotent."""
    combined = f"{source_id}->{hub_id}"
    h = hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]
    return f"mentions_{h}"
