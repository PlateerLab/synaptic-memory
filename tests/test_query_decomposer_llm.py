"""Unit tests for LLMChainDecomposer — offline (stub LLM)."""

from __future__ import annotations

import pytest

from synaptic.extensions.query_decomposer_llm import LLMChainDecomposer


class _StubLLM:
    """Returns a preset response — no network I/O."""

    def __init__(self, response: str) -> None:
        self._response = response

    async def generate(self, *, system: str, user: str, max_tokens: int = 256) -> str:
        return self._response


class _RaisingLLM:
    async def generate(self, *, system: str, user: str, max_tokens: int = 256) -> str:
        raise RuntimeError("upstream down")


@pytest.mark.asyncio
async def test_parses_well_formed_json():
    llm = _StubLLM('{"sub_queries": ["Who distributed UHF?", "UHF distributor founder"]}')
    dec = LLMChainDecomposer(llm=llm)
    out = await dec.decompose("Who founded the company that distributed UHF?")
    assert out == ["Who distributed UHF?", "UHF distributor founder"]


@pytest.mark.asyncio
async def test_malformed_json_falls_back_to_original():
    llm = _StubLLM("not a json")
    dec = LLMChainDecomposer(llm=llm)
    out = await dec.decompose("some query")
    assert out == ["some query"]


@pytest.mark.asyncio
async def test_json_without_sub_queries_key_falls_back():
    llm = _StubLLM('{"other_key": ["a", "b"]}')
    dec = LLMChainDecomposer(llm=llm)
    out = await dec.decompose("some query")
    assert out == ["some query"]


@pytest.mark.asyncio
async def test_empty_array_falls_back_to_original():
    llm = _StubLLM('{"sub_queries": []}')
    dec = LLMChainDecomposer(llm=llm)
    out = await dec.decompose("some query")
    assert out == ["some query"]


@pytest.mark.asyncio
async def test_dedupes_and_strips():
    llm = _StubLLM('{"sub_queries": ["  a  ", "a", "b", "  ", null, "c"]}')
    dec = LLMChainDecomposer(llm=llm)
    out = await dec.decompose("q")
    assert out == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_max_subs_caps_output():
    llm = _StubLLM('{"sub_queries": ["a", "b", "c", "d", "e"]}')
    dec = LLMChainDecomposer(llm=llm, max_subs=2)
    out = await dec.decompose("q")
    assert out == ["a", "b"]


@pytest.mark.asyncio
async def test_upstream_error_returns_original():
    dec = LLMChainDecomposer(llm=_RaisingLLM())
    out = await dec.decompose("some query")
    assert out == ["some query"]


@pytest.mark.asyncio
async def test_empty_query_returns_empty_list():
    llm = _StubLLM('{"sub_queries": ["irrelevant"]}')
    dec = LLMChainDecomposer(llm=llm)
    out = await dec.decompose("   ")
    assert out == []
