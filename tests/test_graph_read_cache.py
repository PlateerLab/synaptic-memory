"""Tests for GraphReadCache helpers."""

from __future__ import annotations

from synaptic.extensions.graph_read_cache import GraphReadCache
from synaptic.models import Edge, EdgeKind


async def test_selective_light_fallback_strips_only_light_kind_properties():
    class FilterOnlyBackend:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[str, ...], str, tuple[str, ...]]] = []

        async def get_edges_batch_filtered(
            self,
            node_ids: list[str],
            *,
            direction: str = "both",
            kinds: list[str | EdgeKind],
        ) -> dict[str, list[Edge]]:
            self.calls.append(
                (tuple(node_ids), direction, tuple(sorted(str(kind) for kind in kinds)))
            )
            return {
                "seed": [
                    Edge(
                        id="related",
                        source_id="seed",
                        target_id="related",
                        kind=EdgeKind.RELATED,
                        properties={"confidence": "0.6"},
                    ),
                    Edge(
                        id="depends",
                        source_id="seed",
                        target_id="depends",
                        kind=EdgeKind.DEPENDS_ON,
                        properties={"confidence": "0.9"},
                    ),
                ]
            }

    backend = FilterOnlyBackend()
    reads = GraphReadCache(backend)  # type: ignore[arg-type]

    mixed = await reads.get_edges_many_by_kind_selective_light(
        ["seed"],
        light_kinds=[EdgeKind.RELATED],
        full_kinds=[EdgeKind.DEPENDS_ON],
    )

    by_id = {edge.id: edge for edge in mixed["seed"]}
    assert backend.calls == [(("seed",), "both", (str(EdgeKind.DEPENDS_ON), str(EdgeKind.RELATED)))]
    assert by_id["related"].properties == {}
    assert by_id["depends"].properties["confidence"] == "0.9"
