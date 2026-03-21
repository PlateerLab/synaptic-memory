"""PostgreSQL backend stub — Phase C (v0.2.0)."""

from __future__ import annotations


class PostgreSQLBackend:
    """PostgreSQL backend with AGE + pgvector + pg_trgm.

    Not yet implemented. Planned for synaptic-memory v0.2.0.
    See docs for the full PostgreSQL schema.
    """

    def __init__(self, dsn: str = "") -> None:
        msg = (
            "PostgreSQL backend is not yet implemented. "
            "Use MemoryBackend or SQLiteBackend for now. "
            "Planned for synaptic-memory v0.2.0."
        )
        raise NotImplementedError(msg)
