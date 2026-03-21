"""Markdown exporter for knowledge graph."""

from __future__ import annotations

from synaptic.models import Node
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
