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


def _row_to_natural_language(table_name: str, row: dict[str, Any]) -> str:
    """Convert a table row to natural language for FTS/vector search.

    Example: {"name": "운동화A", "price": 89000} →
             "테이블 product 행: name=운동화A, price=89000"
    """
    parts = [f"{k}={v}" for k, v in row.items() if v is not None]
    return f"테이블 {table_name} 행: {', '.join(parts)}"


def _row_title(table_name: str, row: dict[str, Any], primary_key: str) -> str:
    """Generate a title for a row node."""
    pk_val = row.get(primary_key, "")
    # Try to find a more descriptive column (name, title, etc.)
    for col in ("name", "title", "label", "이름", "제목"):
        if col in row and row[col]:
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

        Returns:
            List of created ENTITY nodes.
        """
        foreign_keys = foreign_keys or {}
        base_tags = list(tags) if tags else []
        base_tags.append(f"_table:{table_name}")

        # Step 1: Register table schema in ontology (if available)
        self._register_schema(graph, table_name, columns)

        # Step 2: Create nodes for each row
        nodes: list[Node] = []
        chunk_entity_index: ChunkEntityIndex | None = getattr(
            graph, "_chunk_entity_index", None
        )

        for row in rows:
            title = _row_title(table_name, row, primary_key)
            content = _row_to_natural_language(table_name, row)

            # Store all column values as properties
            properties: dict[str, str] = {
                str(k): str(v) for k, v in row.items() if v is not None
            }
            properties["_table_name"] = table_name
            properties["_primary_key"] = primary_key

            node = await graph.add(
                title=title,
                content=content,
                kind=NodeKind.ENTITY,
                tags=list(base_tags),
                source=source,
                properties=properties,
            )
            nodes.append(node)

            # Cache for FK resolution
            pk_val = row.get(primary_key)
            if pk_val is not None:
                self._node_cache[(table_name, str(pk_val))] = node.id

            # Register in chunk-entity index
            if chunk_entity_index is not None:
                chunk_entity_index.register(node.id, node.id)

        # Step 3: Create FK edges
        for row, node in zip(rows, nodes):
            for fk_col, (target_table, target_col) in foreign_keys.items():
                fk_val = row.get(fk_col)
                if fk_val is None:
                    continue

                target_key = (target_table, str(fk_val))
                target_node_id = self._node_cache.get(target_key)

                if target_node_id is not None:
                    await graph.link(
                        node.id,
                        target_node_id,
                        kind=EdgeKind.RELATED,
                        weight=0.8,
                    )
                else:
                    logger.debug(
                        f"FK target not found: {target_table}.{target_col}={fk_val}"
                    )

        logger.info(
            f"Ingested table '{table_name}': {len(nodes)} rows, "
            f"{len(foreign_keys)} FK definitions"
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
