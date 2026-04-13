"""Table data ingester — structured data (DB tables) to knowledge graph.

Converts relational table data into graph nodes and edges:
  - Table schema → OntologyRegistry TypeDef (auto-registration)
  - Each row → ENTITY node (properties store column values)
  - Foreign keys → edges between row nodes
  - Row content auto-converted to natural language for FTS/vector search

No LLM needed — schema provides the structure directly.

Example::

    ingester = TableIngester()
    nodes = await ingester.ingest(
        graph,
        table_name="product",
        columns=[
            {"name": "id", "type": "int"},
            {"name": "name", "type": "str"},
            {"name": "price", "type": "int"},
            {"name": "category_id", "type": "int"},
        ],
        rows=[
            {"id": 1, "name": "운동화A", "price": 89000, "category_id": 3},
            {"id": 2, "name": "티셔츠B", "price": 35000, "category_id": 5},
        ],
        foreign_keys={"category_id": ("category", "id")},
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from synaptic.models import EdgeKind, Node, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
    from synaptic.graph import SynapticGraph
    from synaptic.ontology import OntologyRegistry

logger = logging.getLogger("table-ingester")


# Column-name patterns that typically hold searchable/semantic content.
# Ordered by priority — earlier patterns carry more weight during content
# construction (they get placed at the front of the natural-language row).
# Fully domain-agnostic: the same hints work across Korean, English, and
# multi-language data.
_SEMANTIC_COL_PATTERNS: tuple[tuple[str, ...], ...] = (
    # name / title columns — primary identifiers
    ("name", "title", "label", "nm", "이름", "제목", "상품명", "goods_nm", "product_name"),
    # description / detail columns — rich semantic content
    ("description", "desc", "detail", "content", "설명", "상세", "goods_detail", "product_desc"),
    # category / type / classification columns
    ("category", "type", "kind", "class", "group", "종류", "분류", "category_name", "group_name"),
    # tag / season / attribute columns
    ("tag", "tags", "season", "attribute", "속성", "태그"),
)


def _column_priority(col_name: str) -> int:
    """Return semantic priority for a column (lower = higher priority).

    Used to order values in the natural-language content so that
    name/description/category values appear first, giving them more
    weight in both BM25 FTS (earlier tokens) and embedding models
    (which typically attend more to the start of the input).
    """
    col_lower = col_name.lower()
    for priority, patterns in enumerate(_SEMANTIC_COL_PATTERNS):
        for pat in patterns:
            if pat in col_lower:
                return priority
    return len(_SEMANTIC_COL_PATTERNS)  # unknown columns go last


def _row_to_natural_language(
    table_name: str,
    row: dict[str, Any],
    *,
    column_hints: dict[str, int] | None = None,
) -> str:
    """Convert a table row to value-centric text for FTS/vector search.

    Values are ordered by semantic priority so that descriptive columns
    (name, description, category) appear first. Meta columns starting
    with ``_`` and empty/zero values are excluded.

    Args:
        table_name: Table name, prepended as a prefix.
        row: Dict of column → value.
        column_hints: Optional explicit priority override
            (column_name → priority, lower = earlier). Merged with
            auto-detected priorities; explicit hints win.

    Example: {"name": "운동화A", "price": 89000, "desc": "여름용 가벼운"} →
             "product: 운동화A | 여름용 가벼운 | 89000"
    """
    column_hints = column_hints or {}
    scored: list[tuple[int, str]] = []
    for k, v in row.items():
        if v is None or k.startswith("_"):
            continue
        s = str(v).strip()
        if not s or s in ("0", "0.0", "None", "null"):
            continue
        priority = column_hints.get(k, _column_priority(k))
        scored.append((priority, s))

    scored.sort(key=lambda x: x[0])
    values = [s for _, s in scored]
    return f"{table_name}: {' | '.join(values)}"


def _row_title(table_name: str, row: dict[str, Any], primary_key: str) -> str:
    """Generate a title for a row node."""
    pk_val = row.get(primary_key, "")
    # Try to find a more descriptive column (name, title, etc.)
    for col in ("name", "title", "label", "이름", "제목"):
        if row.get(col):
            return f"{table_name}:{row[col]}"
    return f"{table_name}:{pk_val}"


class TableIngester:
    """Ingests relational table data into the knowledge graph.

    Handles:
    - Schema → OntologyRegistry type registration
    - Rows → ENTITY nodes with properties
    - Foreign keys → edges between nodes
    - Natural language content for search

    Usage::

        ingester = TableIngester()
        nodes = await ingester.ingest(graph, "product", columns, rows,
                                       foreign_keys={"category_id": ("category", "id")})
    """

    __slots__ = ("_node_cache",)

    def __init__(self) -> None:
        # (table_name, pk_value) → node_id for FK resolution
        self._node_cache: dict[tuple[str, str], str] = {}

    async def ingest(
        self,
        graph: SynapticGraph,
        table_name: str,
        columns: list[dict[str, str]],
        rows: list[dict[str, Any]],
        *,
        foreign_keys: dict[str, tuple[str, str]] | None = None,
        primary_key: str = "id",
        tags: list[str] | None = None,
        source: str = "",
        source_url: str = "",
    ) -> list[Node]:
        """Ingest a table into the graph.

        Args:
            graph: SynapticGraph instance.
            table_name: Table name (used for TypeDef, node titles).
            columns: Column definitions [{"name": "col", "type": "str"}, ...].
            rows: Row data [{"col": value, ...}, ...].
            foreign_keys: FK mappings {"col": ("target_table", "target_col")}.
            primary_key: Primary key column name.
            tags: Additional tags for all nodes.
            source: Source identifier.
            source_url: When non-empty, switches to **deterministic node IDs**
                derived from ``(source_url, table_name, pk_value)`` so that
                re-ingesting the same row produces the same node ID. Required
                for CDC / incremental sync. Leave empty for the legacy
                random-UUID behaviour (current default for one-shot ingest).

        Returns:
            List of created ENTITY nodes.
        """
        foreign_keys = foreign_keys or {}
        base_tags = list(tags) if tags else []
        base_tags.append(f"_table:{table_name}")

        # Step 1: Register table schema in ontology (if available)
        self._register_schema(graph, table_name, columns)

        # Lazy import — avoid circular dep with synaptic.extensions.cdc
        deterministic = bool(source_url)
        if deterministic:
            from synaptic.extensions.cdc.ids import (
                canonical_pk,
                deterministic_row_id,
            )
        else:
            # Fallback so the legacy random-UUID path still has a
            # consistent cache-key normaliser. Importing here keeps
            # the module's own dependency graph unchanged.
            from synaptic.extensions.cdc.ids import canonical_pk

        # Step 2: Create nodes for each row
        nodes: list[Node] = []
        chunk_entity_index: ChunkEntityIndex | None = getattr(graph, "_chunk_entity_index", None)

        for row in rows:
            title = _row_title(table_name, row, primary_key)
            content = _row_to_natural_language(table_name, row)

            # Store all column values as properties
            properties: dict[str, str] = {str(k): str(v) for k, v in row.items() if v is not None}
            properties["_table_name"] = table_name
            properties["_primary_key"] = primary_key
            if source_url:
                properties["_source_url"] = source_url

            pk_val = row.get(primary_key)
            node_id: str | None = None
            if deterministic and pk_val is not None:
                node_id = deterministic_row_id(source_url, table_name, pk_val)

            node = await graph.add(
                title=title,
                content=content,
                kind=NodeKind.ENTITY,
                tags=list(base_tags),
                source=source,
                properties=properties,
                node_id=node_id,
            )
            nodes.append(node)

            # Cache for FK resolution
            if pk_val is not None:
                self._node_cache[(table_name, canonical_pk(pk_val))] = node.id

            # Register in chunk-entity index
            if chunk_entity_index is not None:
                chunk_entity_index.register(node.id, node.id)

        # Step 3: Create FK edges
        for row, node in zip(rows, nodes):
            for fk_col, (target_table, target_col) in foreign_keys.items():
                fk_val = row.get(fk_col)
                if fk_val is None:
                    continue

                target_key = (target_table, canonical_pk(fk_val))
                target_node_id = self._node_cache.get(target_key)
                # CDC mode: even if the target is in a different ingest
                # call (different TableIngester instance), we can still
                # reach it because the deterministic ID is derivable.
                if target_node_id is None and deterministic:
                    target_node_id = deterministic_row_id(source_url, target_table, fk_val)

                if target_node_id is not None:
                    await graph.link(
                        node.id,
                        target_node_id,
                        kind=EdgeKind.RELATED,
                        weight=0.8,
                    )
                else:
                    logger.debug(f"FK target not found: {target_table}.{target_col}={fk_val}")

        logger.info(
            f"Ingested table '{table_name}': {len(nodes)} rows, {len(foreign_keys)} FK definitions"
        )
        return nodes

    def _register_schema(
        self,
        graph: SynapticGraph,
        table_name: str,
        columns: list[dict[str, str]],
    ) -> None:
        """Register table schema as OntologyRegistry TypeDef."""
        ontology: OntologyRegistry | None = graph.ontology
        if ontology is None:
            return

        # Check if already registered
        existing = ontology.get_type(table_name)
        if existing is not None:
            return

        from synaptic.ontology import PropertyDef, TypeDef

        props = []
        for col in columns:
            col_name = col.get("name", "")
            col_type = col.get("type", "str")
            if not col_name:
                continue
            # Map SQL types to ontology types
            type_map = {
                "int": "int",
                "integer": "int",
                "float": "float",
                "double": "float",
                "bool": "bool",
                "boolean": "bool",
                "str": "str",
                "string": "str",
                "text": "str",
                "varchar": "str",
            }
            value_type = type_map.get(col_type.lower(), "str")
            props.append(PropertyDef(name=col_name, value_type=value_type))

        typedef = TypeDef(
            name=table_name,
            parent="entity",
            description=f"Table: {table_name}",
            properties=props,
        )

        try:
            ontology.register_type(typedef)
            logger.info(f"Registered ontology type '{table_name}' with {len(props)} properties")
        except ValueError:
            pass  # Already registered or parent not found
