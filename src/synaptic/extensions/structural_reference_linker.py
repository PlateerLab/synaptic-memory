"""StructuralReferenceLinker — LLM-free cross-reference edges (v0.24 WS-A).

Generalises the finreg-specific reference linker into a profile-driven,
corpus-agnostic pass. When a :class:`DomainProfile` declares a clean
target inventory, this writes ``REFERENCES`` edges between documents that
explicitly cite one another — turning multi-hop "follow the citation"
retrieval into a single graph hop (measured: finreg multi-hop 0% → 83%).

Why a clean-target gate
-----------------------
The v0.23 ``ReferenceLinker`` was a *measured negative*: on KRRA its
resolution targets were 70k noisy phrase-hub ENTITY nodes, so mapping a
reference token to the document it points at was a coin flip (~50%
precision). The mechanism was never wrong — the corpus was. A corpus
where every document carries a canonical, low-collision key (statute
article numbers, clause codes, manual section ids) is the opposite case.

This linker therefore *verifies* the target inventory is clean before
writing any edge: if the configured key property collides across
documents beyond a small tolerance, it gates itself off and writes
nothing. That makes the pass safe to run on any corpus — it enriches the
graph where it can and no-ops where it can't.

Configuration (DomainProfile):
    reference_key_property    — node property holding the canonical key
    reference_scope_property  — optional; resolve within this scope only
    reference_token_pattern   — OPTIONAL regex override

The matcher is **auto-derived from the corpus** — there is no need to
hand-write a regex per corpus. The linker reads every distinct value of
``reference_key_property`` actually present in the graph and builds an
exact alternation matcher from those strings (longest-first, so
"제30조의2" wins over "제30조"). A citation is whatever literally equals
a real key. ``reference_token_pattern`` exists only as an override for
the rare case where citations appear in a different surface form than
the stored key.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synaptic.models import Edge, EdgeKind

if TYPE_CHECKING:
    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("structural-reference-linker")

# Max share of key values that may collide across documents before the
# target inventory is deemed too noisy to resolve against.
_MAX_COLLISION_RATE = 0.10


@dataclass(slots=True)
class ReferenceLinkStats:
    """Outcome of one :meth:`StructuralReferenceLinker.link` run."""

    nodes_scanned: int = 0
    keyed_nodes: int = 0
    edges_created: int = 0
    raw_matches: int = 0
    unresolved: int = 0
    collision_rate: float = 0.0
    gated: bool = False
    gate_reason: str = ""


class StructuralReferenceLinker:
    """Profile-driven cross-reference edge builder."""

    __slots__ = ("_profile",)

    def __init__(self, profile: DomainProfile) -> None:
        self._profile = profile

    def _enabled(self) -> bool:
        # Only the key property is required — the matcher is derived from
        # the corpus's own key values, so no hand-written regex is needed.
        return bool(self._profile.reference_key_property)

    async def link(self, backend: StorageBackend) -> ReferenceLinkStats:
        """Scan every node, resolve reference tokens, write REFERENCES edges.

        Returns stats describing what happened — including ``gated=True``
        when the target inventory failed the cleanliness check.
        """
        stats = ReferenceLinkStats()
        if not self._enabled():
            stats.gated = True
            stats.gate_reason = "profile declares no reference_key_property"
            return stats

        p = self._profile
        key_prop = p.reference_key_property
        scope_prop = p.reference_scope_property

        nodes = await backend.list_nodes(kind=None, limit=500_000)
        stats.nodes_scanned = len(nodes)

        # --- Build the target index and measure collisions ---
        index: dict[tuple[str, str], list[str]] = {}
        for n in nodes:
            props = n.properties or {}
            key = props.get(key_prop)
            if not key:
                continue
            scope = props.get(scope_prop, "") if scope_prop else ""
            index.setdefault((scope, key), []).append(n.id)

        stats.keyed_nodes = sum(len(v) for v in index.values())
        if stats.keyed_nodes == 0:
            stats.gated = True
            stats.gate_reason = f"no node carries property {key_prop!r}"
            return stats

        collided = sum(len(v) - 1 for v in index.values() if len(v) > 1)
        stats.collision_rate = collided / stats.keyed_nodes
        if stats.collision_rate > _MAX_COLLISION_RATE:
            stats.gated = True
            stats.gate_reason = (
                f"target inventory too noisy: {stats.collision_rate:.0%} of "
                f"{key_prop!r} values collide (> {_MAX_COLLISION_RATE:.0%})"
            )
            logger.info("StructuralReferenceLinker gated — %s", stats.gate_reason)
            return stats

        # Unambiguous targets only: (scope, key) -> single node_id.
        resolved_index = {k: v[0] for k, v in index.items() if len(v) == 1}

        # Matcher — derived from the corpus's own key values (exact
        # alternation, longest-first so "제30조의2" wins over "제30조").
        # A hand-written reference_token_pattern overrides this.
        if p.reference_token_pattern is not None:
            matcher = p.reference_token_pattern
        else:
            all_keys = sorted({k for _scope, k in index}, key=len, reverse=True)
            matcher = re.compile("|".join(re.escape(k) for k in all_keys))

        # --- Scan node text, resolve references, build edges ---
        edges: list[Edge] = []
        seen: set[tuple[str, str]] = set()
        for n in nodes:
            props = n.properties or {}
            scope = props.get(scope_prop, "") if scope_prop else ""
            for m in matcher.finditer(n.content or ""):
                token = m.group(0)
                stats.raw_matches += 1
                target_id = resolved_index.get((scope, token))
                if target_id is None:
                    stats.unresolved += 1
                    continue
                if target_id == n.id:
                    continue  # self-reference
                key = (n.id, target_id)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    Edge(
                        source_id=n.id,
                        target_id=target_id,
                        kind=EdgeKind.REFERENCES,
                        weight=1.0,
                    )
                )

        if edges:
            await backend.save_edges_batch(edges)
        stats.edges_created = len(edges)
        logger.info(
            "StructuralReferenceLinker: %d REFERENCES edges (raw=%d unresolved=%d)",
            stats.edges_created,
            stats.raw_matches,
            stats.unresolved,
        )
        return stats
