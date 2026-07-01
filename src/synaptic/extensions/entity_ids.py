"""Deterministic IDs for entity hub nodes.

OpenIE, spaCy NER, and the DF-filtered ``EntityLinker`` must land on the
same hub for the same canonical entity. Keeping the helper here avoids
each extractor inventing its own hash namespace.
"""

from __future__ import annotations

import hashlib
import unicodedata

from synaptic.models import EdgeKind


def canonical_entity_text(text: str) -> str:
    """Normalize an entity title for stable hashing and cache lookups."""
    return unicodedata.normalize("NFC", text.strip())


def deterministic_entity_id(canonical: str, *, entity_type: str = "entity") -> str:
    """Return the shared deterministic hub id for an entity.

    The hash input includes a type namespace so future typed inventories
    can intentionally separate homographs. The current extractors pass
    the shared ``"entity"`` namespace by default so phrase hubs, spaCy
    entities, and OpenIE entities collapse when their canonical title is
    identical.
    """
    normalized = canonical_entity_text(canonical)
    type_key = canonical_entity_text(entity_type).lower() or "entity"
    h = hashlib.md5(f"{type_key}\x00{normalized}".encode()).hexdigest()[:16]
    return f"ent_{h}"


def deterministic_edge_id(
    prefix: str,
    source_id: str,
    target_id: str,
    kind: str | EdgeKind,
) -> str:
    """Stable edge id for idempotent extractor/linker writes."""
    combined = f"{source_id}\x00{target_id}\x00{kind!s}"
    h = hashlib.md5(combined.encode()).hexdigest()[:16]
    return f"{prefix}_{h}"


def deterministic_mention_edge_id(source_id: str, hub_id: str) -> str:
    """Shared id for ``source -> ENTITY`` MENTIONS edges."""
    return deterministic_edge_id("mentions", source_id, hub_id, EdgeKind.MENTIONS)
