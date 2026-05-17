"""Tests for ReferenceLinker — connective-pattern typed-edge extraction."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.reference_linker import (
    ReferenceLinker,
    _is_clean_target,
    _ref_edge_id,
    _resolve_in_window,
)
from synaptic.models import ConsolidationLevel, EdgeKind, Node, NodeKind


async def _seed(
    backend: MemoryBackend,
    *,
    targets: dict[str, str],
    chunks: list[str],
) -> None:
    """Seed target nodes (id -> title) and CHUNK source nodes."""
    for nid, title in targets.items():
        await backend.save_node(
            Node(id=nid, kind=NodeKind.RULE, title=title, content=""),
        )
    for i, text in enumerate(chunks):
        await backend.save_node(
            Node(
                id=f"chunk_{i:03d}",
                kind=NodeKind.CHUNK,
                title=f"doc #{i}",
                content=text,
                level=ConsolidationLevel.L0_RAW,
            )
        )


# --- Deterministic ids & helpers ---------------------------------------


class TestHelpers:
    def test_ref_edge_id_stable(self):
        a = _ref_edge_id(EdgeKind.DEPENDS_ON, "chunk_001", "rule_a")
        b = _ref_edge_id(EdgeKind.DEPENDS_ON, "chunk_001", "rule_a")
        assert a == b
        assert a.startswith("ref_")

    def test_ref_edge_id_kind_sensitive(self):
        assert _ref_edge_id(EdgeKind.CAUSED, "x", "y") != _ref_edge_id(
            EdgeKind.DEPENDS_ON, "x", "y"
        )

    def test_clean_target_accepts_entity_names(self):
        assert _is_clean_target("개인정보 보호법") is True
        assert _is_clean_target("직제규정") is True
        assert _is_clean_target("회의") is True  # noun ending in 의

    def test_clean_target_rejects_grammatical_fragments(self):
        assert _is_clean_target("등에 대하여") is False
        assert _is_clean_target("당사자로 하는") is False
        assert _is_clean_target("소상공인으로서") is False
        assert _is_clean_target("a1") is False  # no Hangul
        assert _is_clean_target("") is False

    def test_resolve_in_window_longest_closest(self):
        clean = {"안전관리규정": "rule_a", "규정": "rule_b"}
        # both titles present — longest wins
        assert _resolve_in_window("본 절차는 안전관리규정", clean) == "rule_a"
        # only the short title present
        assert _resolve_in_window("기타 규정", clean) == "rule_b"
        # nothing present
        assert _resolve_in_window("관계 없는 문장", clean) is None


# --- Locale gate -------------------------------------------------------


class TestLocaleGate:
    @pytest.mark.asyncio
    async def test_non_korean_locale_skips(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "개인정보 보호법"},
            chunks=["개인정보 보호법에 따라 처리한다."],
        )
        profile = DomainProfile(name="t", locale="en")
        stats = await ReferenceLinker(profile).link(backend)
        assert stats.skipped_locale is True
        assert stats.edges_created == 0

    @pytest.mark.asyncio
    async def test_multi_locale_runs(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "개인정보 보호법"},
            chunks=["개인정보 보호법에 따라 처리한다."],
        )
        profile = DomainProfile(name="t", locale="multi")
        stats = await ReferenceLinker(profile).link(backend)
        assert stats.skipped_locale is False
        assert stats.edges_created == 1


# --- Connective → EdgeKind mapping -------------------------------------


class TestConnectivePatterns:
    @pytest.mark.asyncio
    async def test_depends_on(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "안전관리규정"},
            chunks=["본 절차는 안전관리규정에 의거하여 수행된다."],
        )
        await ReferenceLinker(DomainProfile(name="t", locale="ko")).link(backend)
        edges = await backend.get_edges("chunk_000", direction="outgoing")
        assert len(edges) == 1
        assert edges[0].kind == EdgeKind.DEPENDS_ON
        assert edges[0].target_id == "rule_a"

    @pytest.mark.asyncio
    async def test_caused(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "설비 노후화"},
            chunks=["설비 노후화로 인해 사고가 발생했다."],
        )
        await ReferenceLinker(DomainProfile(name="t", locale="ko")).link(backend)
        edges = await backend.get_edges("chunk_000", direction="outgoing")
        assert [e.kind for e in edges] == [EdgeKind.CAUSED]

    @pytest.mark.asyncio
    async def test_supersedes(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "구 운영지침"},
            chunks=["이 문서는 구 운영지침을 대체하여 시행된다."],
        )
        await ReferenceLinker(DomainProfile(name="t", locale="ko")).link(backend)
        edges = await backend.get_edges("chunk_000", direction="outgoing")
        assert [e.kind for e in edges] == [EdgeKind.SUPERSEDES]

    @pytest.mark.asyncio
    async def test_contradicts(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "기존 방침"},
            chunks=["기존 방침과 달리 새 절차를 적용한다."],
        )
        await ReferenceLinker(DomainProfile(name="t", locale="ko")).link(backend)
        edges = await backend.get_edges("chunk_000", direction="outgoing")
        assert [e.kind for e in edges] == [EdgeKind.CONTRADICTS]


# --- Target resolution -------------------------------------------------


class TestResolution:
    @pytest.mark.asyncio
    async def test_unresolved_span_dropped(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "개인정보 보호법"},
            chunks=["존재하지않는규정에 따라 처리한다."],
        )
        stats = await ReferenceLinker(DomainProfile(name="t", locale="ko")).link(
            backend
        )
        assert stats.edges_created == 0
        assert stats.unresolved >= 1

    @pytest.mark.asyncio
    async def test_no_self_edge(self):
        """A chunk whose own title matches must not link to itself."""
        backend = MemoryBackend()
        await backend.save_node(
            Node(id="rule_a", kind=NodeKind.RULE, title="공통 절차", content="")
        )
        await backend.save_node(
            Node(
                id="chunk_000",
                kind=NodeKind.CHUNK,
                title="공통 절차",
                content="공통 절차에 따라 진행한다.",
                level=ConsolidationLevel.L0_RAW,
            )
        )
        await ReferenceLinker(DomainProfile(name="t", locale="ko")).link(backend)
        edges = await backend.get_edges("chunk_000", direction="outgoing")
        assert all(e.target_id != "chunk_000" for e in edges)


# --- Idempotency & caps ------------------------------------------------


class TestIdempotencyAndCaps:
    @pytest.mark.asyncio
    async def test_rerun_is_idempotent(self):
        backend = MemoryBackend()
        await _seed(
            backend,
            targets={"rule_a": "안전관리규정"},
            chunks=["절차는 안전관리규정에 따라 수행된다."],
        )
        linker = ReferenceLinker(DomainProfile(name="t", locale="ko"))
        await linker.link(backend)
        await linker.link(backend)
        edges = await backend.get_edges("chunk_000", direction="outgoing")
        assert len(edges) == 1

    @pytest.mark.asyncio
    async def test_per_kind_cap(self):
        backend = MemoryBackend()
        targets = {f"rule_{i}": f"규정{i}번" for i in range(8)}
        sentence = " ".join(f"규정{i}번에 따라" for i in range(8)) + " 처리한다."
        await _seed(backend, targets=targets, chunks=[sentence])
        linker = ReferenceLinker(
            DomainProfile(name="t", locale="ko"), max_per_kind_per_source=3
        )
        await linker.link(backend)
        edges = await backend.get_edges("chunk_000", direction="outgoing")
        assert len(edges) == 3


# --- Empty corpus ------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_no_targets_no_edges(self):
        backend = MemoryBackend()
        await backend.save_node(
            Node(
                id="chunk_000",
                kind=NodeKind.CHUNK,
                title="doc",
                content="안전관리규정에 따라 수행된다.",
                level=ConsolidationLevel.L0_RAW,
            )
        )
        stats = await ReferenceLinker(DomainProfile(name="t", locale="ko")).link(
            backend
        )
        assert stats.edges_created == 0
        assert stats.target_index_size == 0
