"""Helpers for reading structured metadata off a ``Node.properties`` dict.

Synaptic-memory stores temporal / version / authority metadata in the
``Node.properties`` dict rather than as first-class dataclass fields,
because the schema changes more often than the dataclass, and because
every backend already serialises ``properties`` as a JSON blob —
there's no migration cost to adding a new key.

That decision has one downside: every caller that wants to ask "is
this document current? is it authoritative?" has to reach into the
dict by string key. This module centralises those reads so they all
use the same key names, the same type coercions, and the same
fallback behaviour.

Key conventions:

- ``valid_from`` — integer year or unix timestamp at which the
  document became effective. Stored as a string. Absent means
  "always valid" (e.g. a timeless concept like a glossary entry).
- ``valid_to`` — integer year or unix timestamp at which the
  document stopped being current. Absent means "still current".
- ``version`` — free-text version label ("v1.0", "2024-1분기").
- ``authority`` — integer 0-10. Higher means more authoritative.
  Typically set from :attr:`DomainProfile.authority_by_kind`.

Example::

    from synaptic.extensions.node_metadata import (
        authority_of,
        is_current,
        year_of,
    )

    if is_current(node):
        score += authority_of(node)

The helpers are intentionally forgiving — an unparseable year or
missing field returns a sensible default instead of raising. That
matches the pipeline's "prefer partial data over crash" philosophy.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synaptic.models import Node


def _props(node: Node) -> dict[str, str]:
    return node.properties or {}


# --- Temporal ---


def year_of(node: Node) -> int | None:
    """Return the ``year`` property as an int, or ``None`` if missing.

    Checks ``year`` first (set by ``DocumentIngester`` from
    ``DocumentRecord.year``), then falls back to ``valid_from`` which
    we also store as the year when the source only provides year-level
    granularity.
    """
    props = _props(node)
    for key in ("year", "valid_from"):
        raw = props.get(key)
        if raw:
            try:
                return int(str(raw))
            except ValueError:
                continue
    return None


def valid_from_of(node: Node) -> int | None:
    """Return the ``valid_from`` property as an int, or ``None``."""
    raw = _props(node).get("valid_from")
    if not raw:
        return None
    try:
        return int(str(raw))
    except ValueError:
        return None


def valid_to_of(node: Node) -> int | None:
    """Return the ``valid_to`` property as an int, or ``None``."""
    raw = _props(node).get("valid_to")
    if not raw:
        return None
    try:
        return int(str(raw))
    except ValueError:
        return None


def is_current(node: Node, *, as_of: int | None = None) -> bool:
    """Is this node current relative to ``as_of`` (or "now")?

    "Current" means: ``as_of`` is on or after the node's ``valid_from``
    AND on or before its ``valid_to`` (or there is no ``valid_to``).
    Nodes without any temporal markers are considered current too —
    timeless facts don't age.

    ``as_of`` is interpreted the same way as the stored fields: when
    both sides are 4-digit years we compare years; when both sides are
    unix timestamps we compare timestamps. Pass ``None`` to use the
    current year.
    """
    vf = valid_from_of(node)
    vt = valid_to_of(node)
    if vf is None and vt is None:
        return True

    if as_of is None:
        as_of = int(time.strftime("%Y"))

    if vf is not None and as_of < vf:
        return False
    if vt is not None and as_of > vt:
        return False
    return True


# --- Authority ---


def authority_of(node: Node, *, default: int = 0) -> int:
    """Return the ``authority`` property as an int.

    Default is ``0`` (unknown / lowest trust). Callers that want a
    non-zero default for unknown authorities can pass ``default=5`` to
    treat them as "neutral" instead of "untrusted".
    """
    raw = _props(node).get("authority")
    if not raw:
        return default
    try:
        return int(str(raw))
    except ValueError:
        return default


# --- Version ---


def version_of(node: Node) -> str:
    """Return the ``version`` property as a string, empty if missing."""
    return str(_props(node).get("version", ""))


# --- Ranking helper ---


def authority_ranked(nodes: list[Node]) -> list[Node]:
    """Sort nodes by (authority desc, updated_at desc).

    This is the stock ranking for "which of these conflicting sources
    should I trust first?" The agent tools use it when returning
    multiple candidates for the same fact.
    """
    return sorted(
        nodes,
        key=lambda n: (-authority_of(n), -(n.updated_at or 0.0)),
    )
