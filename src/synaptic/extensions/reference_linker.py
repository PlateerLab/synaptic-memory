"""Connective-pattern semantic edge extractor — LLM-free post-pass.

Walks an already-ingested graph, scans source nodes (typically
``NodeKind.CHUNK``) for Korean discourse connectives, and emits *typed*
semantic edges between existing nodes:

- ``…에 따라 / 의거하여``        → ``EdgeKind.DEPENDS_ON``
- ``…(으)로 인해 / 때문에``      → ``EdgeKind.CAUSED``
- ``…를 개정 / 대체 / 폐지하고`` → ``EdgeKind.SUPERSEDES``
- ``…와 달리 / 에 반하여``       → ``EdgeKind.CONTRADICTS``

Unlike :class:`~synaptic.extensions.entity_linker.EntityLinker`, this
linker creates **no new nodes** — it only connects nodes that already
exist. Edge ids are deterministic so re-running on the same corpus is
idempotent (upsert backends overwrite rather than duplicate).

Resolution strategy (v2 — window + clean dictionary)
----------------------------------------------------
A connective alone is not enough; the *referenced node* must be found.
Lazy regex capture of "the phrase before the connective" produced short
fragments that substring-matched noise. Instead this linker:

1. Builds a **clean target dictionary** — titles of candidate nodes,
   filtered to drop grammatical fragments (titles ending in verb /
   particle suffixes such as ``…하는`` / ``…대하여`` / ``…으로서``).
2. For each connective match, scans the **window** of text immediately
   before it and resolves to the *longest, closest* clean title that
   appears verbatim in that window. No match in the window → no edge.

This is an opt-in post-pass, not part of ``from_data()``. The default
ingest path stays LLM-free and zero-cost. Korean connectives are
explicit morphemes, so rule-based precision is high; the linker is
locale-gated and no-ops on non-Korean corpora.

Design: ``docs/PLAN-v0.23-reference-linker.md``.

Example::

    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.extensions.reference_linker import ReferenceLinker

    profile = DomainProfile.load("profiles/my_corpus.toml")
    linker = ReferenceLinker(profile)
    stats = await linker.link(backend, source_kind=NodeKind.CHUNK)
    print(stats.edges_created, stats.by_kind)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from synaptic.models import Edge, EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.protocols import StorageBackend

logger = logging.getLogger(__name__)

# Characters before the connective to inspect for a referenced title.
_WINDOW = 40
# Title length bounds for a usable resolution target.
_MIN_TARGET_TITLE = 2
_MAX_TARGET_TITLE = 40
_EDGE_WEIGHT = 0.7

# Built-in Korean connective regexes (the connective itself, no capture
# group). For each match the *window* of text before ``match.start()``
# is searched for a clean target title. Applied only when the profile
# locale is ``ko`` or ``multi``.
_DEFAULT_KO_CONNECTIVES: dict[EdgeKind, tuple[str, ...]] = {
    EdgeKind.DEPENDS_ON: (
        r"에 (?:따라|의거하여|의거해|근거하여|근거해)",
        r"을(?:를)? 준용",
    ),
    EdgeKind.CAUSED: (
        r"(?:으로|로) 인(?:해|하여)",
        r" 때문에",
        r"의 결과(?:로)?",
        r"에 기인",
    ),
    EdgeKind.SUPERSEDES: (
        r"을(?:를)? (?:개정|대체하여|폐지하고|갈음)",
    ),
    EdgeKind.CONTRADICTS: (
        r"(?:와|과) 달리",
        r"에 반(?:하여|해)",
        r"(?:와|과) 배치",
    ),
}

# Node kinds eligible as the *target* of a reference edge. EntityLinker
# phrase hubs are ENTITY-kind, so they are covered here.
_TARGET_KINDS: tuple[NodeKind, ...] = (
    NodeKind.ENTITY,
    NodeKind.RULE,
    NodeKind.DECISION,
    NodeKind.CONCEPT,
    NodeKind.OBSERVATION,
)

# Multi-char verb / adjective / particle endings. A node title ending in
# any of these is a grammatical fragment, not an entity name, and is
# excluded from the resolution dictionary. Only unambiguous multi-char
# suffixes are listed — single-char endings risk rejecting real nouns
# (회의, 평가, 신한 …).
_BAD_TARGET_SUFFIXES: tuple[str, ...] = (
    "하는", "되는", "하여", "되어", "하고", "되고", "하며", "되며",
    "하게", "되게", "하지", "되지", "하면", "되면", "드는",
    "대하여", "위하여", "관하여", "의하여", "대한", "위한", "관한",
    "의한", "따른", "있는", "없는", "같은", "라는", "다는",
    "으로서", "로서", "에게서", "에게", "에서", "으로",
    "처럼", "보다", "부터", "까지", "만큼", "이고", "이며",
)

_WS = re.compile(r"\s+")
_HANGUL = re.compile(r"[가-힣]")


@dataclass(slots=True)
class ReferenceLinkStats:
    """Outcome of a :meth:`ReferenceLinker.link` run."""

    source_nodes_scanned: int = 0
    target_index_size: int = 0
    target_candidates_seen: int = 0
    raw_matches: int = 0
    resolved: int = 0
    unresolved: int = 0
    edges_created: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    skipped_locale: bool = False


def _ref_edge_id(kind: EdgeKind, source_id: str, target_id: str) -> str:
    """Deterministic edge id so repeated ``link()`` runs are idempotent."""
    combined = f"{kind.value}:{source_id}->{target_id}"
    h = hashlib.blake2b(combined.encode("utf-8"), digest_size=8).hexdigest()
    return f"ref_{h}"


def _is_clean_target(title: str) -> bool:
    """True if a node title is a usable resolution target.

    Rejects empty / out-of-range titles, titles with no Hangul, and
    grammatical fragments ending in a verb / particle suffix.
    """
    t = title.strip()
    if not (_MIN_TARGET_TITLE <= len(t) <= _MAX_TARGET_TITLE):
        return False
    if not _HANGUL.search(t):
        return False
    return not any(t.endswith(suf) for suf in _BAD_TARGET_SUFFIXES)


class ReferenceLinker:
    """Connective-pattern typed-edge extractor (post-pass, LLM-free).

    Args:
        profile: Supplies ``locale`` — the linker no-ops unless it is
            ``ko`` or ``multi``.
        max_per_kind_per_source: Cap on edges of a single kind emitted
            from one source node — suppresses runaway matches.
    """

    __slots__ = ("_compiled", "_max_per_kind", "_profile")

    def __init__(
        self,
        profile: DomainProfile,
        *,
        max_per_kind_per_source: int = 5,
    ) -> None:
        self._profile = profile
        self._max_per_kind = max_per_kind_per_source
        self._compiled: list[tuple[EdgeKind, re.Pattern[str]]] = []
        for kind, pats in _DEFAULT_KO_CONNECTIVES.items():
            for pat in pats:
                try:
                    self._compiled.append((kind, re.compile(pat)))
                except re.error:  # pragma: no cover - static patterns
                    logger.warning("ReferenceLinker: bad pattern skipped: %s", pat)

    async def link(
        self,
        backend: StorageBackend,
        *,
        source_kind: str | NodeKind = NodeKind.CHUNK,
        source_limit: int = 1_000_000,
    ) -> ReferenceLinkStats:
        """Scan sources, match connectives, emit typed edges.

        Args:
            backend: Graph backend to read from and write to.
            source_kind: Which node kind to scan for connectives.
            source_limit: Safety fuse on listing.

        Returns:
            :class:`ReferenceLinkStats` describing the run.
        """
        stats = ReferenceLinkStats()
        t0 = time.time()

        # Locale gate — connectives are only reliable for Korean text.
        if self._profile.locale not in ("ko", "multi"):
            stats.skipped_locale = True
            stats.elapsed_seconds = time.time() - t0
            return stats

        # --- Pass 1: build the clean target dictionary -----------------
        # title_lower -> node_id (first writer wins).
        clean: dict[str, str] = {}
        for kind in _TARGET_KINDS:
            for node in await backend.list_nodes(kind=kind, limit=source_limit):
                stats.target_candidates_seen += 1
                title = node.title.strip()
                if not _is_clean_target(title):
                    continue
                clean.setdefault(title.lower(), node.id)
        stats.target_index_size = len(clean)
        if not clean:
            stats.elapsed_seconds = time.time() - t0
            return stats

        # --- Pass 2 + 3: scan, match, resolve, emit --------------------
        sources = await backend.list_nodes(kind=source_kind, limit=source_limit)
        stats.source_nodes_scanned = len(sources)
        now = time.time()
        new_edges: list[Edge] = []

        for src in sources:
            text = f"{src.title}\n{src.content}" if src.title else src.content
            if not text.strip():
                continue
            lowered = text.lower()
            per_kind: dict[EdgeKind, int] = {}
            seen: set[tuple[EdgeKind, str]] = set()

            for kind, rx in self._compiled:
                for match in rx.finditer(lowered):
                    stats.raw_matches += 1
                    window = lowered[max(0, match.start() - _WINDOW) : match.start()]
                    target_id = _resolve_in_window(window, clean)
                    if target_id is None or target_id == src.id:
                        stats.unresolved += 1
                        continue
                    key = (kind, target_id)
                    if key in seen:
                        continue
                    if per_kind.get(kind, 0) >= self._max_per_kind:
                        continue
                    seen.add(key)
                    per_kind[kind] = per_kind.get(kind, 0) + 1
                    stats.resolved += 1
                    stats.by_kind[kind.value] = stats.by_kind.get(kind.value, 0) + 1
                    new_edges.append(
                        Edge(
                            id=_ref_edge_id(kind, src.id, target_id),
                            source_id=src.id,
                            target_id=target_id,
                            kind=kind,
                            weight=_EDGE_WEIGHT,
                            created_at=now,
                        )
                    )

        await backend.save_edges_batch(new_edges)
        stats.edges_created = len(new_edges)
        stats.elapsed_seconds = time.time() - t0
        return stats


def _resolve_in_window(window: str, clean: dict[str, str]) -> str | None:
    """Return the node id of the longest, closest clean title in a window.

    The window is the (already lowercased) run of text immediately before
    a connective. Scans longest substrings first and, for each length,
    rightmost-first — so the longest title nearest the connective wins.
    Returns ``None`` when no clean title is present (drop the edge).
    """
    w = _WS.sub(" ", window)
    n = len(w)
    if n < _MIN_TARGET_TITLE:
        return None
    upper = min(n, _MAX_TARGET_TITLE)
    for length in range(upper, _MIN_TARGET_TITLE - 1, -1):
        for i in range(n - length, -1, -1):
            hit = clean.get(w[i : i + length])
            if hit is not None:
                return hit
    return None
