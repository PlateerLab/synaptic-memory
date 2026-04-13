"""Change Data Capture — keep the graph in sync with a live database.

Components
----------

- :mod:`synaptic.extensions.cdc.ids` — deterministic ``node_id`` derivation
  from ``(source_url, table, primary_key)`` so re-ingesting the same row
  produces the same node and ``ON CONFLICT DO UPDATE`` works as upsert
  across runs.
- :mod:`synaptic.extensions.cdc.state` — ``SyncStateStore`` wrapping the
  ``syn_cdc_state`` and ``syn_cdc_pk_index`` tables that persist
  watermark, FK snapshots, and the ``(source_url, table, pk) → node_id``
  mapping needed for incremental sync and delete detection.
- :mod:`synaptic.extensions.cdc.sync` — ``TableSyncer`` orchestrating
  timestamp / hash / full strategies (later phases).
- :mod:`synaptic.extensions.cdc.hashing` — row-content hashing for the
  ``hash`` change-detection fallback (later phases).

Public API surface lives on :class:`synaptic.SynapticGraph`:

.. code-block:: python

    # Initial CDC load (deterministic IDs + sync state)
    graph = await SynapticGraph.from_database(dsn, mode="cdc")

    # Subsequent incremental sync
    result = await graph.sync_from_database(dsn)
    print(result.added, result.updated, result.deleted)
"""

from synaptic.extensions.cdc.ids import (
    deterministic_row_id,
    normalize_source_url,
)
from synaptic.extensions.cdc.state import SyncStateStore

__all__ = [
    "SyncStateStore",
    "deterministic_row_id",
    "normalize_source_url",
]
