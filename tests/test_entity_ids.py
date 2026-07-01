"""Shared deterministic entity ids across extractors."""

from __future__ import annotations

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.entity_extractor_spacy import SpaCyEntityExtractor
from synaptic.extensions.entity_ids import deterministic_entity_id
from synaptic.graph import SynapticGraph
from synaptic.models import Node, NodeKind


class _Ent:
    text = "Acme"
    label_ = "ORG"
    start_char = 0
    end_char = 4


class _Doc:
    def __init__(self) -> None:
        self.ents = [_Ent()]


class _FakeNLP:
    def __call__(self, text):
        return _Doc()


def _fake_spacy_extractor() -> SpaCyEntityExtractor:
    extractor = SpaCyEntityExtractor.__new__(SpaCyEntityExtractor)
    extractor._max_entities = 15
    extractor._min_entity_len = 2
    extractor._fallback = None
    extractor._entity_cache = {}
    extractor._ko_nlp = None
    extractor._en_nlp = _FakeNLP()
    return extractor


@pytest.mark.asyncio
async def test_spacy_extractor_uses_shared_deterministic_entity_id():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    await backend.save_node(Node(id="chunk_1", kind=NodeKind.CHUNK, title="Acme", content=""))
    extractor = _fake_spacy_extractor()

    ids = await extractor.extract_and_link(graph, "chunk_1", "Acme", "")

    expected = deterministic_entity_id("Acme")
    assert ids == [expected]
    node = await backend.get_node(expected)
    assert node is not None
    assert "_spacy" in node.tags
    assert "_label:ORG" in node.tags
