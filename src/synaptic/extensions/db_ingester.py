"""DbIngester — auto-ingest from a relational database into a knowledge graph.

Connects to a database, auto-discovers schema (tables, columns, types,
foreign keys), and ingests all data into a synaptic-memory graph. No
manual schema specification needed — just a connection string.

Supported databases:
- SQLite (built-in, no extra deps)
- PostgreSQL (requires asyncpg: ``pip install synaptic-memory[postgresql]``)

Usage::

    from synaptic.extensions.db_ingester import DbIngester

    ingester = DbIngester()
    stats = await ingester.ingest_from_sqlite("my_database.db", backend)

    # Or via SynapticGraph easy API:
    graph = await SynapticGraph.from_database("sqlite:///my_database.db")
    graph = await SynapticGraph.from_database("postgresql://user:pass@host/db")
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from synaptic.extensions.table_ingester import TableIngester
from synaptic.models import NodeKind

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph

logger = logging.getLogger("db-ingester")


@dataclass(slots=True)
class ColumnInfo:
    name: str
    type: str  # "int", "float", "str", "date", etc.
    primary_key: bool = False
    nullable: bool = True


@dataclass(slots=True)
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass(slots=True)
class TableSchema:
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: str = ""
    foreign_keys: list[ForeignKey] = field(default_factory=list)


@dataclass(slots=True)
class DbIngestStats:
    tables_discovered: int = 0
    tables_ingested: int = 0
    total_rows: int = 0
    total_nodes: int = 0
    total_fk_edges: int = 0
    elapsed_seconds: float = 0.0


# --- SQLite schema reader ---


def _read_sqlite_schema(db_path: str) -> list[TableSchema]:
    """Read schema from a SQLite database using PRAGMA."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Get all tables (skip sqlite internal tables)
    tables_raw = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()

    schemas: list[TableSchema] = []
    for t in tables_raw:
        table_name = t["name"]

        # Columns
        cols_raw = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        columns = []
        pk = ""
        for c in cols_raw:
            col_type = _sqlite_type_map(c["type"])
            is_pk = bool(c["pk"])
            columns.append(ColumnInfo(
                name=c["name"], type=col_type,
                primary_key=is_pk, nullable=not c["notnull"],
            ))
            if is_pk and not pk:
                pk = c["name"]

        # Foreign keys
        fks_raw = con.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
        fks = []
        for fk in fks_raw:
            fks.append(ForeignKey(
                column=fk["from"],
                ref_table=fk["table"],
                ref_column=fk["to"],
            ))

        schemas.append(TableSchema(
            name=table_name, columns=columns,
            primary_key=pk or (columns[0].name if columns else "id"),
            foreign_keys=fks,
        ))

    con.close()
    return schemas


def _sqlite_type_map(raw_type: str) -> str:
    raw = raw_type.upper()
    if "INT" in raw:
        return "int"
    if "REAL" in raw or "FLOAT" in raw or "DOUBLE" in raw or "NUMERIC" in raw:
        return "float"
    if "BOOL" in raw:
        return "bool"
    return "str"


def _read_sqlite_rows(db_path: str, table: str, limit: int = 500_000) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"SELECT * FROM [{table}] LIMIT ?", (limit,)).fetchall()
    result = [dict(r) for r in rows]
    con.close()
    return result


# --- PostgreSQL schema reader ---


async def _read_pg_schema(dsn: str) -> list[TableSchema]:
    """Read schema from PostgreSQL using information_schema."""
    try:
        import asyncpg
    except ImportError as exc:
        msg = "pip install asyncpg for PostgreSQL support"
        raise ImportError(msg) from exc

    con = await asyncpg.connect(dsn)

    # Tables
    tables_raw = await con.fetch("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)

    schemas: list[TableSchema] = []
    for t in tables_raw:
        table_name = t["table_name"]

        # Columns
        cols_raw = await con.fetch("""
            SELECT column_name, data_type, is_nullable,
                   column_default
            FROM information_schema.columns
            WHERE table_name = $1 AND table_schema = 'public'
            ORDER BY ordinal_position
        """, table_name)

        columns = []
        for c in cols_raw:
            columns.append(ColumnInfo(
                name=c["column_name"],
                type=_pg_type_map(c["data_type"]),
                nullable=c["is_nullable"] == "YES",
            ))

        # Primary key
        pk_raw = await con.fetch("""
            SELECT a.attname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = $1::regclass AND i.indisprimary
        """, table_name)
        pk = pk_raw[0]["attname"] if pk_raw else (columns[0].name if columns else "id")

        # Foreign keys
        fks_raw = await con.fetch("""
            SELECT kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.table_name = $1 AND tc.constraint_type = 'FOREIGN KEY'
        """, table_name)

        fks = [ForeignKey(
            column=fk["column_name"],
            ref_table=fk["ref_table"],
            ref_column=fk["ref_column"],
        ) for fk in fks_raw]

        schemas.append(TableSchema(
            name=table_name, columns=columns,
            primary_key=pk, foreign_keys=fks,
        ))

    await con.close()
    return schemas


def _pg_type_map(raw_type: str) -> str:
    raw = raw_type.lower()
    if raw in ("integer", "bigint", "smallint", "serial", "bigserial"):
        return "int"
    if raw in ("real", "double precision", "numeric", "decimal"):
        return "float"
    if raw in ("boolean",):
        return "bool"
    if "timestamp" in raw or "date" in raw:
        return "date"
    return "str"


async def _read_pg_rows(dsn: str, table: str, limit: int = 500_000) -> list[dict]:
    import asyncpg
    con = await asyncpg.connect(dsn)
    rows = await con.fetch(f'SELECT * FROM "{table}" LIMIT $1', limit)
    result = [dict(r) for r in rows]
    await con.close()
    return result


# --- DbIngester ---


class DbIngester:
    """Auto-ingest a relational database into a knowledge graph.

    Discovers schema, reads data, and calls TableIngester for each table.
    Foreign keys become typed RELATED edges automatically.
    """

    __slots__ = ("_table_ingester", "_batch_size")

    def __init__(self, *, batch_size: int = 10_000) -> None:
        self._table_ingester = TableIngester()
        self._batch_size = batch_size

    async def ingest_from_sqlite(
        self,
        db_path: str,
        graph: SynapticGraph,
        *,
        tables: list[str] | None = None,
        row_limit: int = 500_000,
    ) -> DbIngestStats:
        """Ingest from a SQLite database file."""
        t0 = time.time()
        schemas = _read_sqlite_schema(db_path)
        if tables:
            schemas = [s for s in schemas if s.name in tables]

        stats = DbIngestStats(tables_discovered=len(schemas))

        # Ingest order: tables without FKs first, then tables with FKs
        no_fk = [s for s in schemas if not s.foreign_keys]
        has_fk = [s for s in schemas if s.foreign_keys]
        ordered = no_fk + has_fk

        for schema in ordered:
            rows = _read_sqlite_rows(db_path, schema.name, limit=row_limit)
            if not rows:
                continue

            col_defs = [{"name": c.name, "type": c.type} for c in schema.columns]
            fk_map = {fk.column: (fk.ref_table, fk.ref_column) for fk in schema.foreign_keys}

            # Cast values
            cast_rows = [_cast_row(r, schema.columns) for r in rows]

            nodes = await self._table_ingester.ingest(
                graph, schema.name, col_defs, cast_rows,
                primary_key=schema.primary_key,
                foreign_keys=fk_map if fk_map else None,
            )
            stats.tables_ingested += 1
            stats.total_rows += len(rows)
            stats.total_nodes += len(nodes)
            stats.total_fk_edges += len(fk_map) * len(rows)
            logger.info("Ingested %s: %d rows → %d nodes", schema.name, len(rows), len(nodes))

        stats.elapsed_seconds = time.time() - t0
        return stats

    async def ingest_from_postgres(
        self,
        dsn: str,
        graph: SynapticGraph,
        *,
        tables: list[str] | None = None,
        row_limit: int = 500_000,
    ) -> DbIngestStats:
        """Ingest from a PostgreSQL database."""
        t0 = time.time()
        schemas = await _read_pg_schema(dsn)
        if tables:
            schemas = [s for s in schemas if s.name in tables]

        stats = DbIngestStats(tables_discovered=len(schemas))

        no_fk = [s for s in schemas if not s.foreign_keys]
        has_fk = [s for s in schemas if s.foreign_keys]
        ordered = no_fk + has_fk

        for schema in ordered:
            rows = await _read_pg_rows(dsn, schema.name, limit=row_limit)
            if not rows:
                continue

            col_defs = [{"name": c.name, "type": c.type} for c in schema.columns]
            fk_map = {fk.column: (fk.ref_table, fk.ref_column) for fk in schema.foreign_keys}

            cast_rows = [_cast_row(r, schema.columns) for r in rows]

            nodes = await self._table_ingester.ingest(
                graph, schema.name, col_defs, cast_rows,
                primary_key=schema.primary_key,
                foreign_keys=fk_map if fk_map else None,
            )
            stats.tables_ingested += 1
            stats.total_rows += len(rows)
            stats.total_nodes += len(nodes)
            stats.total_fk_edges += len(fk_map) * len(rows)
            logger.info("Ingested %s: %d rows → %d nodes", schema.name, len(rows), len(nodes))

        stats.elapsed_seconds = time.time() - t0
        return stats


def _cast_row(row: dict, columns: list[ColumnInfo]) -> dict:
    """Cast row values to proper Python types."""
    result = {}
    col_types = {c.name: c.type for c in columns}
    for k, v in row.items():
        if v is None:
            continue
        t = col_types.get(k, "str")
        try:
            if t == "int":
                result[k] = int(v)
            elif t == "float":
                result[k] = float(v)
            elif t == "bool":
                result[k] = bool(v)
            else:
                result[k] = str(v)
        except (ValueError, TypeError):
            result[k] = str(v)
    return result
