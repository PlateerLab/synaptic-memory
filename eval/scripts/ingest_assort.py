"""Ingest assort structured (CSV) data into a synaptic-memory graph.

Converts 10 relational CSV tables into a knowledge graph using
TableIngester. Foreign-key relationships become RELATED edges.

Ingest order respects FK dependencies:
  colors, sizes, sales_channels, sales_partners
  → products
  → product_variants (FK: product_code, color_id, size_id)
  → orders (FK: product_code, channel_id, partner_id)
  → reviews (FK: product_code)
  → broadcasts (FK: product_code)
  → variant_sales (FK: variant_code)

Usage::

    uv run python eval/scripts/ingest_assort.py
    uv run python eval/scripts/ingest_assort.py --backend kuzu
    uv run python eval/scripts/ingest_assort.py --clean
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.extensions.table_ingester import TableIngester
from synaptic.graph import SynapticGraph

DATA_DIR = REPO_ROOT / "eval" / "data" / "raw" / "assort"
DEFAULT_SQLITE = REPO_ROOT / "eval" / "data" / "assort_graph.sqlite"
DEFAULT_KUZU = REPO_ROOT / "eval" / "data" / "assort_graph.kuzu"


def _read_csv(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.csv"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _col_defs(rows: list[dict]) -> list[dict[str, str]]:
    """Auto-detect column types from first row."""
    if not rows:
        return []
    type_map: dict[str, str] = {}
    for key, val in rows[0].items():
        try:
            int(val)
            type_map[key] = "int"
        except (ValueError, TypeError):
            try:
                float(val)
                type_map[key] = "float"
            except (ValueError, TypeError):
                type_map[key] = "str"
    return [{"name": k, "type": v} for k, v in type_map.items()]


def _cast_row(row: dict, col_defs: list[dict]) -> dict:
    """Cast row values to proper Python types."""
    result = {}
    type_map = {c["name"]: c["type"] for c in col_defs}
    for k, v in row.items():
        if not v:
            continue
        t = type_map.get(k, "str")
        try:
            if t == "int":
                result[k] = int(v)
            elif t == "float":
                result[k] = float(v)
            else:
                result[k] = v
        except (ValueError, TypeError):
            result[k] = v
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--backend", choices=["sqlite", "kuzu"], default="sqlite")
    p.add_argument("--graph", type=Path, default=None)
    p.add_argument("--clean", action="store_true", help="Delete existing graph first")
    return p.parse_args()


async def _open_backend(backend_name: str, graph_path: Path):
    if backend_name == "sqlite":
        from synaptic.backends.sqlite_graph import SqliteGraphBackend

        backend = SqliteGraphBackend(str(graph_path))
        await backend.connect()
        return backend
    from synaptic.backends.kuzu import KuzuBackend

    backend = KuzuBackend(str(graph_path))
    await backend.connect()
    return backend


async def main() -> int:
    args = _parse_args()
    graph_path = args.graph or (DEFAULT_SQLITE if args.backend == "sqlite" else DEFAULT_KUZU)

    if args.clean and graph_path.exists():
        print(f"Cleaning {graph_path} ...")
        if graph_path.is_dir():
            shutil.rmtree(graph_path)
        else:
            graph_path.unlink()

    print(f"Backend: {args.backend}  Graph: {graph_path.relative_to(REPO_ROOT)}")
    print(f"Data:    {DATA_DIR.relative_to(REPO_ROOT)}")

    backend = await _open_backend(args.backend, graph_path)
    graph = SynapticGraph(backend)
    ingester = TableIngester()

    t0 = time.perf_counter()
    total_nodes = 0

    # --- 1. Lookup tables (no FKs) ---
    for table, pk in [
        ("colors", "color_id"),
        ("sizes", "size_id"),
        ("sales_channels", "channel_id"),
        ("sales_partners", "partner_id"),
    ]:
        rows = _read_csv(table)
        cols = _col_defs(rows)
        cast = [_cast_row(r, cols) for r in rows]
        nodes = await ingester.ingest(graph, table, cols, cast, primary_key=pk)
        total_nodes += len(nodes)
        print(f"  {table}: {len(nodes)} rows")

    # --- 2. Products ---
    rows = _read_csv("products")
    cols = _col_defs(rows)
    cast = [_cast_row(r, cols) for r in rows]
    nodes = await ingester.ingest(graph, "products", cols, cast, primary_key="product_code")
    total_nodes += len(nodes)
    print(f"  products: {len(nodes)} rows")

    # --- 3. Product variants (FK → products, colors, sizes) ---
    rows = _read_csv("product_variants")
    cols = _col_defs(rows)
    cast = [_cast_row(r, cols) for r in rows]
    nodes = await ingester.ingest(
        graph,
        "product_variants",
        cols,
        cast,
        primary_key="variant_code",
        foreign_keys={
            "product_code": ("products", "product_code"),
            "color_id": ("colors", "color_id"),
            "size_id": ("sizes", "size_id"),
        },
    )
    total_nodes += len(nodes)
    print(f"  product_variants: {len(nodes)} rows")

    # --- 4. Orders (FK → products, channels, partners) ---
    rows = _read_csv("orders")
    cols = _col_defs(rows)
    cast = [_cast_row(r, cols) for r in rows]
    nodes = await ingester.ingest(
        graph,
        "orders",
        cols,
        cast,
        primary_key="order_id",
        foreign_keys={
            "product_code": ("products", "product_code"),
            "channel_id": ("sales_channels", "channel_id"),
            "partner_id": ("sales_partners", "partner_id"),
        },
    )
    total_nodes += len(nodes)
    print(f"  orders: {len(nodes)} rows")

    # --- 5. Reviews (FK → products) ---
    rows = _read_csv("reviews")
    cols = _col_defs(rows)
    cast = [_cast_row(r, cols) for r in rows]
    nodes = await ingester.ingest(
        graph,
        "reviews",
        cols,
        cast,
        primary_key="review_id",
        foreign_keys={"product_code": ("products", "product_code")},
    )
    total_nodes += len(nodes)
    print(f"  reviews: {len(nodes)} rows")

    # --- 6. Broadcasts (FK → products) ---
    rows = _read_csv("broadcasts")
    cols = _col_defs(rows)
    cast = [_cast_row(r, cols) for r in rows]
    nodes = await ingester.ingest(
        graph,
        "broadcasts",
        cols,
        cast,
        primary_key="broadcast_id",
        foreign_keys={"product_code": ("products", "product_code")},
    )
    total_nodes += len(nodes)
    print(f"  broadcasts: {len(nodes)} rows")

    # --- 7. Variant sales (FK → product_variants) ---
    rows = _read_csv("variant_sales")
    cols = _col_defs(rows)
    cast = [_cast_row(r, cols) for r in rows]
    nodes = await ingester.ingest(
        graph,
        "variant_sales",
        cols,
        cast,
        primary_key="id",
        foreign_keys={"variant_code": ("product_variants", "variant_code")},
    )
    total_nodes += len(nodes)
    print(f"  variant_sales: {len(nodes)} rows")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"Ingest complete — {elapsed:.1f}s")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Graph path:  {graph_path.relative_to(REPO_ROOT)}")
    print(f"{'=' * 60}")

    # Quick search probe
    print("\n[Quick search probe]")
    from synaptic.search import HybridSearch

    searcher = HybridSearch()
    for q in ["실크블렌드 가디건", "소라 색상", "홈쇼핑 방송 일정"]:
        result = await searcher.search(backend, q, limit=3)
        top = result.nodes[0].node.title if result.nodes else "—"
        print(f"  '{q}' → {len(result.nodes)} hits, top: {top[:60]}")

    await backend.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
