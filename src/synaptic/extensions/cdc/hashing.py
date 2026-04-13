"""Row-content hashing used by the ``hash`` change-detection strategy.

When a table has no ``updated_at``-style column, we cannot use a
``WHERE >= watermark`` filter — instead we read every row and skip
the ones whose content hash matches the prior snapshot stored in
``syn_cdc_pk_index.row_hash``.

The hash needs three properties:

1. **Deterministic across processes** — Python's built-in ``hash()``
   is salted and varies per interpreter, so we use blake2b.
2. **Order-independent** — column iteration order from a SQLite
   row may differ run-to-run, so we sort by column name before
   feeding bytes to the hasher.
3. **Cheap** — runs once per row per sync, hot path on tables with
   100k+ rows. blake2b is the fastest stdlib hash on most CPUs.

We deliberately do NOT use ``json.dumps(sort_keys=True)`` — JSON
escaping rules add ambiguity for binary-like values. A plain
``key=repr(value)`` join is enough since we only need stable
*identity*, not human-readable serialisation.
"""

from __future__ import annotations

import hashlib
from typing import Any


def row_hash(
    row: dict[str, Any],
    *,
    exclude: set[str] | None = None,
    digest_size: int = 8,
) -> str:
    """Return a stable, salt-free content hash of ``row``.

    Default 64-bit (16 hex) digest matches the deterministic node
    ID width — collisions inside a single table at typical sizes
    (~1M rows) stay well under birthday-paradox concerns.

    ``exclude`` lets callers drop columns whose presence shouldn't
    invalidate the hash. The change column itself (e.g.
    ``updated_at``) is the obvious candidate when mixing strategies,
    but in pure hash mode we usually pass ``None`` and hash the
    entire row.
    """
    exclude_set = exclude or set()
    items = []
    for key in sorted(row.keys()):
        if key in exclude_set:
            continue
        if key.startswith("_"):
            continue  # synaptic-internal meta cols
        value = row[key]
        if value is None:
            continue
        items.append(f"{key}={value!r}")
    payload = "\x1f".join(items).encode("utf-8", errors="replace")
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()
