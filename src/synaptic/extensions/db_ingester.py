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

import asyncio
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


# --- MySQL/MariaDB schema reader ---


async def _read_mysql_schema(dsn: str) -> list[TableSchema]:
    """Read schema from MySQL/MariaDB using information_schema."""
    try:
        import aiomysql
    except ImportError as exc:
        msg = "pip install aiomysql for MySQL/MariaDB support"
        raise ImportError(msg) from exc

    # Parse DSN: mysql://user:pass@host:port/dbname
    from urllib.parse import urlparse
    parsed = urlparse(dsn)
    db_name = parsed.path.lstrip("/")

    con = await aiomysql.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        db=db_name,
    )
    cur = await con.cursor(aiomysql.DictCursor)

    await cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """, (db_name,))
    tables_raw = await cur.fetchall()

    schemas: list[TableSchema] = []
    for t in tables_raw:
        table_name = t["TABLE_NAME"] if "TABLE_NAME" in t else t["table_name"]

        await cur.execute("""
            SELECT column_name, data_type, is_nullable, column_key
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """, (db_name, table_name))
        cols_raw = await cur.fetchall()

        columns = []
        pk = ""
        for c in cols_raw:
            col_name = c.get("COLUMN_NAME", c.get("column_name", ""))
            col_type = _mysql_type_map(c.get("DATA_TYPE", c.get("data_type", "")))
            col_key = c.get("COLUMN_KEY", c.get("column_key", ""))
            columns.append(ColumnInfo(
                name=col_name, type=col_type,
                primary_key=col_key == "PRI",
                nullable=c.get("IS_NULLABLE", c.get("is_nullable", "YES")) == "YES",
            ))
            if col_key == "PRI" and not pk:
                pk = col_name

        await cur.execute("""
            SELECT column_name, referenced_table_name, referenced_column_name
            FROM information_schema.key_column_usage
            WHERE table_schema = %s AND table_name = %s
              AND referenced_table_name IS NOT NULL
        """, (db_name, table_name))
        fks_raw = await cur.fetchall()

        fks = [ForeignKey(
            column=fk.get("COLUMN_NAME", fk.get("column_name", "")),
            ref_table=fk.get("REFERENCED_TABLE_NAME", fk.get("referenced_table_name", "")),
            ref_column=fk.get("REFERENCED_COLUMN_NAME", fk.get("referenced_column_name", "")),
        ) for fk in fks_raw]

        schemas.append(TableSchema(
            name=table_name, columns=columns,
            primary_key=pk or (columns[0].name if columns else "id"),
            foreign_keys=fks,
        ))

    await cur.close()
    con.close()
    return schemas


def _mysql_type_map(raw_type: str) -> str:
    raw = raw_type.lower()
    if raw in ("int", "bigint", "smallint", "tinyint", "mediumint"):
        return "int"
    if raw in ("float", "double", "decimal", "numeric"):
        return "float"
    if raw in ("tinyint(1)", "boolean", "bool"):
        return "bool"
    if "date" in raw or "timestamp" in raw:
        return "date"
    return "str"


async def _read_mysql_rows(dsn: str, table: str, limit: int = 500_000) -> list[dict]:
    import aiomysql
    from urllib.parse import urlparse
    parsed = urlparse(dsn)
    con = await aiomysql.connect(
        host=parsed.hostname or "localhost",
        port=parsed.port or 3306,
        user=parsed.username or "root",
        password=parsed.password or "",
        db=parsed.path.lstrip("/"),
    )
    cur = await con.cursor(aiomysql.DictCursor)
    await cur.execute(f"SELECT * FROM `{table}` LIMIT %s", (limit,))
    rows = await cur.fetchall()
    await cur.close()
    con.close()
    return [dict(r) for r in rows]


# --- Oracle schema reader ---


async def _read_oracle_schema(dsn: str) -> list[TableSchema]:
    """Read schema from Oracle using ALL_TAB_COLUMNS / ALL_CONSTRAINTS."""
    try:
        import oracledb
    except ImportError as exc:
        msg = "pip install oracledb for Oracle support"
        raise ImportError(msg) from exc

    # dsn: oracle://user:pass@host:port/service_name
    from urllib.parse import urlparse
    parsed = urlparse(dsn)
    con = await oracledb.connect_async(
        user=parsed.username,
        password=parsed.password,
        dsn=f"{parsed.hostname}:{parsed.port or 1521}/{parsed.path.lstrip('/')}",
    )
    owner = (parsed.username or "").upper()

    cur = con.cursor()
    await cur.execute("""
        SELECT table_name FROM all_tables
        WHERE owner = :owner ORDER BY table_name
    """, owner=owner)
    tables_raw = await cur.fetchall()

    schemas: list[TableSchema] = []
    for (table_name,) in tables_raw:
        await cur.execute("""
            SELECT column_name, data_type, nullable
            FROM all_tab_columns
            WHERE owner = :owner AND table_name = :tbl
            ORDER BY column_id
        """, owner=owner, tbl=table_name)
        cols_raw = await cur.fetchall()

        columns = []
        for col_name, data_type, nullable in cols_raw:
            columns.append(ColumnInfo(
                name=col_name, type=_oracle_type_map(data_type),
                nullable=nullable == "Y",
            ))

        # Primary key
        await cur.execute("""
            SELECT cols.column_name FROM all_constraints cons
            JOIN all_cons_columns cols ON cons.constraint_name = cols.constraint_name
            WHERE cons.owner = :owner AND cons.table_name = :tbl
              AND cons.constraint_type = 'P'
        """, owner=owner, tbl=table_name)
        pk_rows = await cur.fetchall()
        pk = pk_rows[0][0] if pk_rows else (columns[0].name if columns else "ID")

        # Foreign keys
        await cur.execute("""
            SELECT a.column_name, c_pk.table_name, b.column_name
            FROM all_cons_columns a
            JOIN all_constraints c ON a.constraint_name = c.constraint_name
            JOIN all_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
            JOIN all_cons_columns b ON c_pk.constraint_name = b.constraint_name
            WHERE c.owner = :owner AND c.table_name = :tbl
              AND c.constraint_type = 'R'
        """, owner=owner, tbl=table_name)
        fks_raw = await cur.fetchall()

        fks = [ForeignKey(column=col, ref_table=ref_tbl, ref_column=ref_col)
               for col, ref_tbl, ref_col in fks_raw]

        schemas.append(TableSchema(
            name=table_name, columns=columns,
            primary_key=pk, foreign_keys=fks,
        ))

    await cur.close()
    await con.close()
    return schemas


def _oracle_type_map(raw_type: str) -> str:
    raw = raw_type.upper()
    if "NUMBER" in raw or "INTEGER" in raw:
        return "float"  # Oracle NUMBER can be int or float
    if "FLOAT" in raw or "DOUBLE" in raw:
        return "float"
    if "DATE" in raw or "TIMESTAMP" in raw:
        return "date"
    return "str"


async def _read_oracle_rows(dsn: str, table: str, limit: int = 500_000) -> list[dict]:
    import oracledb
    from urllib.parse import urlparse
    parsed = urlparse(dsn)
    con = await oracledb.connect_async(
        user=parsed.username,
        password=parsed.password,
        dsn=f"{parsed.hostname}:{parsed.port or 1521}/{parsed.path.lstrip('/')}",
    )
    cur = con.cursor()
    await cur.execute(f'SELECT * FROM "{table}" FETCH FIRST :lim ROWS ONLY', lim=limit)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in await cur.fetchall()]
    await cur.close()
    await con.close()
    return rows


# --- SQL Server schema reader ---


async def _read_mssql_schema(dsn: str) -> list[TableSchema]:
    """Read schema from SQL Server using information_schema."""
    try:
        import aioodbc
    except ImportError as exc:
        msg = "pip install aioodbc for SQL Server support"
        raise ImportError(msg) from exc

    con = await aioodbc.connect(dsn=dsn)
    cur = await con.cursor()

    await cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_type = 'BASE TABLE' AND table_schema = 'dbo'
        ORDER BY table_name
    """)
    tables_raw = await cur.fetchall()

    schemas: list[TableSchema] = []
    for (table_name,) in tables_raw:
        await cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = ? AND table_schema = 'dbo'
            ORDER BY ordinal_position
        """, table_name)
        cols_raw = await cur.fetchall()

        columns = [ColumnInfo(
            name=r[0], type=_mssql_type_map(r[1]),
            nullable=r[2] == "YES",
        ) for r in cols_raw]

        # PK
        await cur.execute("""
            SELECT column_name FROM information_schema.key_column_usage kcu
            JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
            WHERE tc.table_name = ? AND tc.constraint_type = 'PRIMARY KEY'
        """, table_name)
        pk_rows = await cur.fetchall()
        pk = pk_rows[0][0] if pk_rows else (columns[0].name if columns else "id")

        # FK
        await cur.execute("""
            SELECT kcu.column_name, ccu.table_name, ccu.column_name
            FROM information_schema.referential_constraints rc
            JOIN information_schema.key_column_usage kcu
                ON rc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON rc.unique_constraint_name = ccu.constraint_name
            WHERE kcu.table_name = ?
        """, table_name)
        fks_raw = await cur.fetchall()

        fks = [ForeignKey(column=r[0], ref_table=r[1], ref_column=r[2])
               for r in fks_raw]

        schemas.append(TableSchema(
            name=table_name, columns=columns,
            primary_key=pk, foreign_keys=fks,
        ))

    await cur.close()
    await con.close()
    return schemas


def _mssql_type_map(raw_type: str) -> str:
    raw = raw_type.lower()
    if raw in ("int", "bigint", "smallint", "tinyint"):
        return "int"
    if raw in ("float", "real", "decimal", "numeric", "money"):
        return "float"
    if raw in ("bit",):
        return "bool"
    if "date" in raw or "time" in raw:
        return "date"
    return "str"


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

    async def ingest_from_mysql(
        self,
        dsn: str,
        graph: SynapticGraph,
        *,
        tables: list[str] | None = None,
        row_limit: int = 500_000,
    ) -> DbIngestStats:
        """Ingest from MySQL/MariaDB."""
        t0 = time.time()
        schemas = await _read_mysql_schema(dsn)
        if tables:
            schemas = [s for s in schemas if s.name in tables]
        return await self._ingest_schemas(
            schemas, graph,
            row_reader=lambda tbl: _read_mysql_rows(dsn, tbl, limit=row_limit),
            t0=t0,
        )

    async def ingest_from_oracle(
        self,
        dsn: str,
        graph: SynapticGraph,
        *,
        tables: list[str] | None = None,
        row_limit: int = 500_000,
    ) -> DbIngestStats:
        """Ingest from Oracle."""
        t0 = time.time()
        schemas = await _read_oracle_schema(dsn)
        if tables:
            schemas = [s for s in schemas if s.name in tables]
        return await self._ingest_schemas(
            schemas, graph,
            row_reader=lambda tbl: _read_oracle_rows(dsn, tbl, limit=row_limit),
            t0=t0,
        )

    async def ingest_from_mssql(
        self,
        dsn: str,
        graph: SynapticGraph,
        *,
        tables: list[str] | None = None,
        row_limit: int = 500_000,
    ) -> DbIngestStats:
        """Ingest from SQL Server."""
        t0 = time.time()
        schemas = await _read_mssql_schema(dsn)
        if tables:
            schemas = [s for s in schemas if s.name in tables]
        return await self._ingest_schemas(
            schemas, graph,
            row_reader=lambda tbl: _read_mssql_rows(dsn, tbl, limit=row_limit),
            t0=t0,
        )

    async def _ingest_schemas(
        self,
        schemas: list[TableSchema],
        graph: SynapticGraph,
        *,
        row_reader,
        t0: float,
    ) -> DbIngestStats:
        """Shared ingestion logic for all DB types."""
        stats = DbIngestStats(tables_discovered=len(schemas))
        no_fk = [s for s in schemas if not s.foreign_keys]
        has_fk = [s for s in schemas if s.foreign_keys]

        for schema in no_fk + has_fk:
            rows = await row_reader(schema.name) if asyncio.iscoroutinefunction(row_reader) else row_reader(schema.name)
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
