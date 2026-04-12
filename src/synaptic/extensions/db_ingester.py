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
from typing import TYPE_CHECKING

from synaptic.extensions.table_ingester import TableIngester

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
            columns.append(
                ColumnInfo(
                    name=c["name"],
                    type=col_type,
                    primary_key=is_pk,
                    nullable=not c["notnull"],
                )
            )
            if is_pk and not pk:
                pk = c["name"]

        # Foreign keys
        fks_raw = con.execute(f"PRAGMA foreign_key_list('{table_name}')").fetchall()
        fks = []
        for fk in fks_raw:
            fks.append(
                ForeignKey(
                    column=fk["from"],
                    ref_table=fk["table"],
                    ref_column=fk["to"],
                )
            )

        schemas.append(
            TableSchema(
                name=table_name,
                columns=columns,
                primary_key=pk or (columns[0].name if columns else "id"),
                foreign_keys=fks,
            )
        )

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
        cols_raw = await con.fetch(
            """
            SELECT column_name, data_type, is_nullable,
                   column_default
            FROM information_schema.columns
            WHERE table_name = $1 AND table_schema = 'public'
            ORDER BY ordinal_position
        """,
            table_name,
        )

        columns = []
        for c in cols_raw:
            columns.append(
                ColumnInfo(
                    name=c["column_name"],
                    type=_pg_type_map(c["data_type"]),
                    nullable=c["is_nullable"] == "YES",
                )
            )

        # Primary key
        pk_raw = await con.fetch(
            """
            SELECT a.attname FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = $1::regclass AND i.indisprimary
        """,
            table_name,
        )
        pk = pk_raw[0]["attname"] if pk_raw else (columns[0].name if columns else "id")

        # Foreign keys
        fks_raw = await con.fetch(
            """
            SELECT kcu.column_name, ccu.table_name AS ref_table, ccu.column_name AS ref_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON tc.constraint_name = ccu.constraint_name
            WHERE tc.table_name = $1 AND tc.constraint_type = 'FOREIGN KEY'
        """,
            table_name,
        )

        fks = [
            ForeignKey(
                column=fk["column_name"],
                ref_table=fk["ref_table"],
                ref_column=fk["ref_column"],
            )
            for fk in fks_raw
        ]

        schemas.append(
            TableSchema(
                name=table_name,
                columns=columns,
                primary_key=pk,
                foreign_keys=fks,
            )
        )

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

    await cur.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = %s AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """,
        (db_name,),
    )
    tables_raw = await cur.fetchall()

    schemas: list[TableSchema] = []
    for t in tables_raw:
        table_name = t["TABLE_NAME"] if "TABLE_NAME" in t else t["table_name"]

        await cur.execute(
            """
            SELECT column_name, data_type, is_nullable, column_key
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position
        """,
            (db_name, table_name),
        )
        cols_raw = await cur.fetchall()

        columns = []
        pk = ""
        for c in cols_raw:
            col_name = c.get("COLUMN_NAME", c.get("column_name", ""))
            col_type = _mysql_type_map(c.get("DATA_TYPE", c.get("data_type", "")))
            col_key = c.get("COLUMN_KEY", c.get("column_key", ""))
            columns.append(
                ColumnInfo(
                    name=col_name,
                    type=col_type,
                    primary_key=col_key == "PRI",
                    nullable=c.get("IS_NULLABLE", c.get("is_nullable", "YES")) == "YES",
                )
            )
            if col_key == "PRI" and not pk:
                pk = col_name

        await cur.execute(
            """
            SELECT column_name, referenced_table_name, referenced_column_name
            FROM information_schema.key_column_usage
            WHERE table_schema = %s AND table_name = %s
              AND referenced_table_name IS NOT NULL
        """,
            (db_name, table_name),
        )
        fks_raw = await cur.fetchall()

        fks = [
            ForeignKey(
                column=fk.get("COLUMN_NAME", fk.get("column_name", "")),
                ref_table=fk.get("REFERENCED_TABLE_NAME", fk.get("referenced_table_name", "")),
                ref_column=fk.get("REFERENCED_COLUMN_NAME", fk.get("referenced_column_name", "")),
            )
            for fk in fks_raw
        ]

        schemas.append(
            TableSchema(
                name=table_name,
                columns=columns,
                primary_key=pk or (columns[0].name if columns else "id"),
                foreign_keys=fks,
            )
        )

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
    from urllib.parse import urlparse

    import aiomysql

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
    await cur.execute(
        """
        SELECT table_name FROM all_tables
        WHERE owner = :owner ORDER BY table_name
    """,
        owner=owner,
    )
    tables_raw = await cur.fetchall()

    schemas: list[TableSchema] = []
    for (table_name,) in tables_raw:
        await cur.execute(
            """
            SELECT column_name, data_type, nullable
            FROM all_tab_columns
            WHERE owner = :owner AND table_name = :tbl
            ORDER BY column_id
        """,
            owner=owner,
            tbl=table_name,
        )
        cols_raw = await cur.fetchall()

        columns = []
        for col_name, data_type, nullable in cols_raw:
            columns.append(
                ColumnInfo(
                    name=col_name,
                    type=_oracle_type_map(data_type),
                    nullable=nullable == "Y",
                )
            )

        # Primary key
        await cur.execute(
            """
            SELECT cols.column_name FROM all_constraints cons
            JOIN all_cons_columns cols ON cons.constraint_name = cols.constraint_name
            WHERE cons.owner = :owner AND cons.table_name = :tbl
              AND cons.constraint_type = 'P'
        """,
            owner=owner,
            tbl=table_name,
        )
        pk_rows = await cur.fetchall()
        pk = pk_rows[0][0] if pk_rows else (columns[0].name if columns else "ID")

        # Foreign keys
        await cur.execute(
            """
            SELECT a.column_name, c_pk.table_name, b.column_name
            FROM all_cons_columns a
            JOIN all_constraints c ON a.constraint_name = c.constraint_name
            JOIN all_constraints c_pk ON c.r_constraint_name = c_pk.constraint_name
            JOIN all_cons_columns b ON c_pk.constraint_name = b.constraint_name
            WHERE c.owner = :owner AND c.table_name = :tbl
              AND c.constraint_type = 'R'
        """,
            owner=owner,
            tbl=table_name,
        )
        fks_raw = await cur.fetchall()

        fks = [
            ForeignKey(column=col, ref_table=ref_tbl, ref_column=ref_col)
            for col, ref_tbl, ref_col in fks_raw
        ]

        schemas.append(
            TableSchema(
                name=table_name,
                columns=columns,
                primary_key=pk,
                foreign_keys=fks,
            )
        )

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
    from urllib.parse import urlparse

    import oracledb

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
        await cur.execute(
            """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = ? AND table_schema = 'dbo'
            ORDER BY ordinal_position
        """,
            table_name,
        )
        cols_raw = await cur.fetchall()

        columns = [
            ColumnInfo(
                name=r[0],
                type=_mssql_type_map(r[1]),
                nullable=r[2] == "YES",
            )
            for r in cols_raw
        ]

        # PK
        await cur.execute(
            """
            SELECT column_name FROM information_schema.key_column_usage kcu
            JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
            WHERE tc.table_name = ? AND tc.constraint_type = 'PRIMARY KEY'
        """,
            table_name,
        )
        pk_rows = await cur.fetchall()
        pk = pk_rows[0][0] if pk_rows else (columns[0].name if columns else "id")

        # FK
        await cur.execute(
            """
            SELECT kcu.column_name, ccu.table_name, ccu.column_name
            FROM information_schema.referential_constraints rc
            JOIN information_schema.key_column_usage kcu
                ON rc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu
                ON rc.unique_constraint_name = ccu.constraint_name
            WHERE kcu.table_name = ?
        """,
            table_name,
        )
        fks_raw = await cur.fetchall()

        fks = [ForeignKey(column=r[0], ref_table=r[1], ref_column=r[2]) for r in fks_raw]

        schemas.append(
            TableSchema(
                name=table_name,
                columns=columns,
                primary_key=pk,
                foreign_keys=fks,
            )
        )

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


async def _read_mssql_rows(dsn: str, table: str, limit: int = 500_000) -> list[dict]:
    """Read rows from SQL Server via aioodbc."""
    try:
        import aioodbc
    except ImportError as exc:
        msg = "pip install aioodbc for SQL Server support"
        raise ImportError(msg) from exc

    con = await aioodbc.connect(dsn=dsn)
    cur = await con.cursor()
    # SQL Server uses TOP instead of LIMIT
    await cur.execute(f"SELECT TOP {int(limit)} * FROM [{table}]")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in await cur.fetchall()]
    await cur.close()
    await con.close()
    return rows


# --- DbIngester ---


def _is_join_table(schema: TableSchema) -> bool:
    """Detect M:N join/bridge tables.

    A join table typically has:
    - Exactly 2 FK columns (pointing to 2 different tables)
    - At most 1 additional column (PK/ID)
    - No other meaningful data columns

    Example: product_categories(id PK, product_id FK, category_id FK)
    → should become a direct edge, not a node.
    """
    if len(schema.foreign_keys) < 2:
        return False
    fk_cols = {fk.column for fk in schema.foreign_keys}
    non_fk_cols = [c for c in schema.columns if c.name not in fk_cols and not c.primary_key]
    # At most 1 extra column (e.g. a created_at timestamp) is OK
    return len(non_fk_cols) <= 1


class DbIngester:
    """Auto-ingest a relational database into a knowledge graph.

    Discovers schema, reads data, and calls TableIngester for each table.
    Foreign keys become typed RELATED edges automatically.

    **M:N join tables** are detected automatically: tables with 2+ FKs
    and no meaningful data columns become direct edges between the
    referenced tables instead of intermediate nodes.

    **Batch processing**: large tables are read in configurable batches
    (default 10,000 rows) to avoid memory exhaustion.
    """

    __slots__ = ("_batch_size", "_table_ingester")

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
        return await self._ingest_schemas(
            schemas,
            graph,
            row_reader=lambda tbl: _read_sqlite_rows(db_path, tbl, limit=row_limit),
            t0=t0,
        )

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
            schemas,
            graph,
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
            schemas,
            graph,
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
            schemas,
            graph,
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
        """Shared ingestion logic for all DB types.

        Handles M:N join tables as edges and processes large tables
        in batches.
        """
        stats = DbIngestStats(tables_discovered=len(schemas))

        # Separate: regular tables vs M:N join tables
        regular = []
        join_tables = []
        for s in schemas:
            if _is_join_table(s):
                join_tables.append(s)
                logger.info("Detected M:N join table: %s → will create edges", s.name)
            else:
                regular.append(s)

        # Ingest order: tables without FKs first, then FK tables
        no_fk = [s for s in regular if not s.foreign_keys]
        has_fk = [s for s in regular if s.foreign_keys]

        for schema in no_fk + has_fk:
            result = row_reader(schema.name)
            rows = await result if asyncio.iscoroutine(result) else result
            if not rows:
                continue

            col_defs = [{"name": c.name, "type": c.type} for c in schema.columns]
            fk_map = {fk.column: (fk.ref_table, fk.ref_column) for fk in schema.foreign_keys}

            # Batch processing for large tables
            total_nodes = 0
            for i in range(0, len(rows), self._batch_size):
                batch = rows[i : i + self._batch_size]
                cast_batch = [_cast_row(r, schema.columns) for r in batch]
                nodes = await self._table_ingester.ingest(
                    graph,
                    schema.name,
                    col_defs,
                    cast_batch,
                    primary_key=schema.primary_key,
                    foreign_keys=fk_map if fk_map else None,
                )
                total_nodes += len(nodes)

            stats.tables_ingested += 1
            stats.total_rows += len(rows)
            stats.total_nodes += total_nodes
            logger.info("Ingested %s: %d rows → %d nodes", schema.name, len(rows), total_nodes)

        # M:N join tables → direct edges (no intermediate nodes)
        for schema in join_tables:
            result = row_reader(schema.name)
            rows = await result if asyncio.iscoroutine(result) else result
            if not rows:
                continue

            fk_a = schema.foreign_keys[0]
            fk_b = schema.foreign_keys[1]
            edge_count = 0

            from synaptic.models import EdgeKind

            for row in rows:
                val_a = str(row.get(fk_a.column, ""))
                val_b = str(row.get(fk_b.column, ""))
                if not val_a or not val_b:
                    continue

                # Find the nodes by their cached keys
                key_a = (fk_a.ref_table, val_a)
                key_b = (fk_b.ref_table, val_b)
                node_a_id = self._table_ingester._node_cache.get(key_a)
                node_b_id = self._table_ingester._node_cache.get(key_b)

                if node_a_id and node_b_id:
                    await graph.link(
                        node_a_id,
                        node_b_id,
                        kind=EdgeKind.RELATED,
                        weight=0.8,
                    )
                    edge_count += 1

            stats.total_fk_edges += edge_count
            logger.info(
                "M:N %s: %d rows → %d edges (%s ↔ %s)",
                schema.name,
                len(rows),
                edge_count,
                fk_a.ref_table,
                fk_b.ref_table,
            )

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
        return await self._ingest_schemas(
            schemas,
            graph,
            row_reader=lambda tbl: _read_pg_rows(dsn, tbl, limit=row_limit),
            t0=t0,
        )


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
