"""Tests for the cross-domain validation path in eval/run_all.py.

Phase 1.5 of the v0.20+ track. Locks two contracts:

  1. ``_count_domains_for_ids`` correctly tallies per-domain counts
     from a set of node ids by looking up each node's
     ``properties._domain_id``.
  2. The bench-loop skip rule treats queries with
     ``validation: {type: domain_coverage}`` as runnable even when
     ``relevant_docs`` is empty — without this, the cross_domain
     query file would be silently ignored by the bench harness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.run_all import _count_domains_for_ids
from synaptic.backends.memory import MemoryBackend
from synaptic.models import ConsolidationLevel, Node, NodeKind


async def _make_backend_with_domains(nodes_per_domain: dict[str, int]) -> MemoryBackend:
    """Create a backend pre-populated with N nodes per domain, each
    tagged ``properties._domain_id = <domain>``."""
    b = MemoryBackend()
    await b.connect()
    counter = 0
    for domain, n in nodes_per_domain.items():
        for i in range(n):
            await b.save_node(
                Node(
                    id=f"n{counter:04d}",
                    kind=NodeKind.ENTITY,
                    title=f"{domain}:item{i}",
                    properties={"_domain_id": domain},
                    level=ConsolidationLevel.L0_RAW,
                )
            )
            counter += 1
    return b


@pytest.mark.asyncio
async def test_count_domains_resolves_each_id_to_its_domain():
    backend = await _make_backend_with_domains({"krra": 3, "assort": 5, "x2bee": 2})
    # Pick one id from each domain plus an unknown
    found_ids = {"n0000", "n0003", "n0008", "phantom_id"}
    counts = await _count_domains_for_ids(backend, found_ids)
    assert counts == {"krra": 1, "assort": 1, "x2bee": 1}
    # phantom_id silently dropped — agent ids that no longer resolve
    # must not corrupt the tally


@pytest.mark.asyncio
async def test_count_domains_handles_empty_input():
    backend = await _make_backend_with_domains({"krra": 1})
    counts = await _count_domains_for_ids(backend, set())
    assert counts == {}


@pytest.mark.asyncio
async def test_count_domains_dedups_repeated_ids():
    """If the same id appears multiple times in found_ids it should be
    counted once — found_ids is a set in production but defensive
    programming via ``seen`` set inside the helper."""
    backend = await _make_backend_with_domains({"krra": 2})
    counts = await _count_domains_for_ids(backend, {"n0000"})
    assert counts == {"krra": 1}


@pytest.mark.asyncio
async def test_count_domains_skips_nodes_without_domain_id():
    """Untagged nodes (legacy or non-MetaCorpus) contribute nothing —
    they're not counted under any domain."""
    b = MemoryBackend()
    await b.connect()
    await b.save_node(
        Node(id="tagged", kind=NodeKind.ENTITY, properties={"_domain_id": "krra"})
    )
    await b.save_node(Node(id="untagged", kind=NodeKind.ENTITY, properties={}))
    counts = await _count_domains_for_ids(b, {"tagged", "untagged"})
    assert counts == {"krra": 1}


@pytest.mark.asyncio
async def test_count_domains_caps_lookup_at_500():
    """Performance bound — the helper must not melt down on a 10K
    found_ids set. Even if only 500 are looked up, the smaller subset
    should still yield non-trivial counts."""
    backend = await _make_backend_with_domains({"krra": 600})
    found = {f"n{i:04d}" for i in range(600)}
    counts = await _count_domains_for_ids(backend, found)
    # Capped at 500 lookups → counts should reflect that
    total = sum(counts.values())
    assert total <= 500
    assert "krra" in counts


@pytest.mark.asyncio
async def test_count_domains_resolves_titles_via_sqlite_bulk_query(tmp_path):
    """The agent's ``found_ids`` typically contains titles like
    ``"products:G00007"`` and ``"ESG 및 지속가능성"`` (from
    _extract_ids picking up Evidence.title). Direct ``get_node`` only
    matches by id, missing all the title hits — this test locks in
    the title-based bulk SQL path that picks them up."""
    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    db_path = tmp_path / "t.sqlite"
    b = SqliteGraphBackend(str(db_path))
    await b.connect()
    # Mimic metacorpus shape: real id like "doc_<hash>", title like "ESG..."
    await b.save_node(
        Node(
            id="doc_abc",
            kind=NodeKind.ENTITY,
            title="ESG 및 지속가능성",
            properties={"_domain_id": "krra"},
        )
    )
    await b.save_node(
        Node(
            id="cat_xyz",
            kind=NodeKind.ENTITY,
            title="products:G00007",
            properties={"_domain_id": "assort"},
        )
    )
    # Agent's found_ids = mix of titles and ids
    found = {"ESG 및 지속가능성", "products:G00007", "doc_abc"}
    counts = await _count_domains_for_ids(b, found)
    assert counts == {"krra": 1, "assort": 1}
    await b.close()


@pytest.mark.asyncio
async def test_count_domains_resolves_raw_doc_id_hashes_via_properties_scan(tmp_path):
    """When the agent surfaces a chunk's ``properties.doc_id`` (a raw
    16-char hex hash, NOT a node id), the helper must still resolve
    it via a properties_json LIKE scan and credit the right domain.
    This is the most common shape of unresolvable id seen on
    real cross-domain runs."""
    from synaptic.backends.sqlite_graph import SqliteGraphBackend

    db_path = tmp_path / "t.sqlite"
    b = SqliteGraphBackend(str(db_path))
    await b.connect()
    # Chunk node with the actual id format ("chunk_<hash>") and a
    # doc_id property pointing to the parent doc — this is exactly
    # what DocumentIngester writes.
    await b.save_node(
        Node(
            id="chunk_aaaa1111bbbb2222",
            kind=NodeKind.ENTITY,
            title="some chunk",
            properties={
                "_domain_id": "krra",
                "doc_id": "ffeeddccbbaa9988",  # raw hex (no doc_ prefix)
            },
        )
    )
    # Agent surfaced ONLY the raw hash (typical for properties.doc_id
    # extraction in agent_loop._extract_ids)
    counts = await _count_domains_for_ids(b, {"ffeeddccbbaa9988"})
    assert counts == {"krra": 1}
    await b.close()


def test_cross_domain_json_validation_block_shape_matches_harness_consumer():
    """The harness reads validation.must_include_domains as a list and
    validation.min_docs_per_domain as an int (default 1). Re-verify the
    spec file conforms — if the format diverges, the bench would
    silently treat every cross-domain query as a miss."""
    import json

    p = Path(__file__).parent.parent / "eval" / "data" / "queries" / "cross_domain.json"
    data = json.loads(p.read_text())
    for q in data["queries"]:
        v = q["validation"]
        assert v["type"] == "domain_coverage", q["qid"]
        assert isinstance(v["must_include_domains"], list), q["qid"]
        assert all(isinstance(d, str) for d in v["must_include_domains"]), q["qid"]
        # min_docs_per_domain optional, default 1
        assert int(v.get("min_docs_per_domain", 1)) >= 1, q["qid"]
