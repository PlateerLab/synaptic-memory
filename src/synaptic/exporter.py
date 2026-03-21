"""Exporters for knowledge graph — Markdown and JSON."""

from __future__ import annotations

import json

from synaptic.models import Edge, Node
from synaptic.protocols import StorageBackend


class MarkdownExporter:
    """Exports nodes to Markdown format."""

    __slots__ = ()

    async def export(
        self,
        backend: StorageBackend,
        *,
        node_ids: list[str] | None = None,
    ) -> str:
        if node_ids is not None:
            nodes: list[Node] = []
            for nid in node_ids:
                node = await backend.get_node(nid)
                if node is not None:
                    nodes.append(node)
        else:
            nodes = await backend.list_nodes(limit=500)

        if not nodes:
            return "# Knowledge Graph\n\nNo nodes found.\n"

        lines: list[str] = ["# Knowledge Graph\n"]

        # Group by kind
        by_kind: dict[str, list[Node]] = {}
        for node in nodes:
            kind = str(node.kind)
            if kind not in by_kind:
                by_kind[kind] = []
            by_kind[kind].append(node)

        for kind in sorted(by_kind):
            lines.append(f"\n## {kind.title()}\n")
            for node in sorted(by_kind[kind], key=lambda n: n.title):
                tags = ", ".join(node.tags) if node.tags else ""
                tag_suffix = f" [{tags}]" if tags else ""
                lines.append(f"### {node.title}{tag_suffix}\n")
                lines.append(f"- **Level**: {node.level}")
                lines.append(f"- **Vitality**: {node.vitality:.2f}")
                lines.append(
                    f"- **Usage**: {node.access_count} accesses, "
                    f"{node.success_count} success, {node.failure_count} failure"
                )
                if node.source:
                    lines.append(f"- **Source**: {node.source}")
                lines.append(f"\n{node.content}\n")

        return "\n".join(lines)


class JSONExporter:
    """Exports nodes and edges as JSON."""

    __slots__ = ()

    async def export(
        self,
        backend: StorageBackend,
        *,
        node_ids: list[str] | None = None,
    ) -> str:
        if node_ids is not None:
            nodes: list[Node] = []
            for nid in node_ids:
                node = await backend.get_node(nid)
                if node is not None:
                    nodes.append(node)
        else:
            nodes = await backend.list_nodes(limit=500)

        # Collect edges for these nodes
        node_id_set = {n.id for n in nodes}
        edges: list[Edge] = []
        for node in nodes:
            node_edges = await backend.get_edges(node.id, direction="outgoing")
            for edge in node_edges:
                if edge.target_id in node_id_set:
                    edges.append(edge)

        data = {
            "nodes": [
                {
                    "id": n.id,
                    "kind": str(n.kind),
                    "title": n.title,
                    "content": n.content,
                    "tags": n.tags,
                    "level": str(n.level),
                    "vitality": n.vitality,
                    "access_count": n.access_count,
                    "success_count": n.success_count,
                    "failure_count": n.failure_count,
                    "source": n.source,
                }
                for n in nodes
            ],
            "edges": [
                {
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "kind": str(e.kind),
                    "weight": e.weight,
                }
                for e in edges
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)
