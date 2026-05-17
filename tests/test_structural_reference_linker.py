"""Tests for StructuralReferenceLinker (v0.24 WS-A)."""

from __future__ import annotations

import re

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.extensions.structural_reference_linker import StructuralReferenceLinker
from synaptic.models import EdgeKind, Node, NodeKind


def _profile(**kw) -> DomainProfile:
    # No reference_token_pattern — the matcher is auto-derived from the
    # corpus's own article_no values (the default, hardcode-free path).
    base = {
        "name": "t",
        "locale": "ko",
        "reference_key_property": "article_no",
        "reference_scope_property": "law",
    }
    base.update(kw)
    return DomainProfile(**base)


def _article(law: str, no: str, content: str) -> Node:
    return Node(
        kind=NodeKind.ENTITY,
        title=f"{law} {no}",
        content=content,
        properties={"law": law, "article_no": no},
    )


@pytest.mark.asyncio
async def test_clean_inventory_creates_edges():
    backend = MemoryBackend()
    a = _article("은행법", "제3조", "은행은 제5조에 따라 인가를 받아야 한다.")
    b = _article("은행법", "제5조", "제5조(인가) 은행업 인가의 요건은 다음과 같다.")
    await backend.save_node(a)
    await backend.save_node(b)

    stats = await StructuralReferenceLinker(_profile()).link(backend)

    assert not stats.gated
    assert stats.edges_created == 1
    edges = await backend.get_edges(a.id, direction="outgoing")
    refs = [e for e in edges if e.kind == EdgeKind.REFERENCES]
    assert len(refs) == 1
    assert refs[0].target_id == b.id


@pytest.mark.asyncio
async def test_scope_isolates_resolution():
    """A '제5조' citation resolves within the citing node's own law only."""
    backend = MemoryBackend()
    a = _article("은행법", "제3조", "제5조에 따라 처리한다.")
    same_law = _article("은행법", "제5조", "제5조 본문")
    other_law = _article("보험업법", "제5조", "제5조 본문")
    for n in (a, same_law, other_law):
        await backend.save_node(n)

    stats = await StructuralReferenceLinker(_profile()).link(backend)

    assert stats.edges_created == 1
    edges = await backend.get_edges(a.id, direction="outgoing")
    assert [e.target_id for e in edges if e.kind == EdgeKind.REFERENCES] == [same_law.id]


@pytest.mark.asyncio
async def test_self_reference_excluded():
    backend = MemoryBackend()
    a = _article("은행법", "제5조", "제5조(인가) 이 조의 적용범위는 ...")
    await backend.save_node(a)

    stats = await StructuralReferenceLinker(_profile()).link(backend)

    assert stats.edges_created == 0


@pytest.mark.asyncio
async def test_gate_on_noisy_target_inventory():
    """If the key property collides heavily, the linker gates itself off."""
    backend = MemoryBackend()
    # 10 nodes all sharing article_no '제1조' within one law → 90% collision.
    for i in range(10):
        await backend.save_node(
            _article("noisy법", "제1조", f"제2조 참조 본문 {i}")
        )
    await backend.save_node(_article("noisy법", "제2조", "제2조 본문"))

    stats = await StructuralReferenceLinker(_profile()).link(backend)

    assert stats.gated
    assert stats.collision_rate > 0.10
    assert stats.edges_created == 0


@pytest.mark.asyncio
async def test_disabled_profile_is_gated():
    backend = MemoryBackend()
    await backend.save_node(_article("은행법", "제3조", "제5조 참조"))
    stats = await StructuralReferenceLinker(
        DomainProfile(name="t", locale="ko")
    ).link(backend)
    assert stats.gated
    assert stats.edges_created == 0


@pytest.mark.asyncio
async def test_auto_derived_matcher_prefers_longest_key():
    """'제3조의2' must win over '제3조' when both are real article keys."""
    backend = MemoryBackend()
    citing = _article("은행법", "제1조", "제3조의2에 따라 처리한다.")
    short = _article("은행법", "제3조", "제3조 본문")
    longer = _article("은행법", "제3조의2", "제3조의2 본문")
    for n in (citing, short, longer):
        await backend.save_node(n)

    await StructuralReferenceLinker(_profile()).link(backend)

    edges = await backend.get_edges(citing.id, direction="outgoing")
    targets = [e.target_id for e in edges if e.kind == EdgeKind.REFERENCES]
    assert targets == [longer.id]


@pytest.mark.asyncio
async def test_token_pattern_override():
    """An explicit reference_token_pattern overrides the auto-derived matcher."""
    backend = MemoryBackend()
    a = _article("은행법", "제3조", "제5조에 따라 처리한다.")
    b = _article("은행법", "제5조", "제5조 본문")
    await backend.save_node(a)
    await backend.save_node(b)

    profile = _profile(reference_token_pattern=re.compile(r"제\d+조(?:의\d+)?"))
    stats = await StructuralReferenceLinker(profile).link(backend)

    assert stats.edges_created == 1


@pytest.mark.asyncio
async def test_edges_deduplicated():
    backend = MemoryBackend()
    a = _article("은행법", "제3조", "제5조에 따라, 그리고 다시 제5조에 의거하여 처리한다.")
    b = _article("은행법", "제5조", "제5조 본문")
    await backend.save_node(a)
    await backend.save_node(b)

    stats = await StructuralReferenceLinker(_profile()).link(backend)

    assert stats.edges_created == 1  # two citations of 제5조 → one edge
