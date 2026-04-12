"""Tests for DocumentIngester — generic corpus → graph."""

from __future__ import annotations

import json

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.document_ingester import (
    ChunkRecord,
    DocumentIngester,
    DocumentRecord,
    InMemoryDocumentSource,
    JsonlDocumentSource,
)
from synaptic.extensions.domain_profile import DomainProfile
from synaptic.models import NodeKind


def _sample_doc(
    doc_id: str = "d1",
    title: str = "Doc 1",
    category: str = "",
    chunks: list[ChunkRecord] | None = None,
    content: str = "",
) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        title=title,
        content=content,
        category=category,
        chunks=chunks or [],
    )


def _chunk(doc_id: str, index: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=f"{doc_id}_c{index:04d}",
        doc_id=doc_id,
        text=text,
        index=index,
    )


# --- Basic ingest ---


class TestBasicIngest:
    @pytest.mark.asyncio
    async def test_single_doc_no_chunks(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        ingester = DocumentIngester(profile=profile, backend=backend)

        source = InMemoryDocumentSource(
            [_sample_doc("d1", "제목", content="본문")]
        )
        stats = await ingester.ingest(source)

        assert stats.documents_ingested == 1
        assert stats.chunks_created == 0

        docs = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)
        assert any(n.title == "제목" for n in docs)

    @pytest.mark.asyncio
    async def test_doc_with_chunks_creates_contains_and_next_chunk_edges(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        ingester = DocumentIngester(profile=profile, backend=backend)

        doc = _sample_doc(
            "d1",
            "Doc 1",
            chunks=[
                _chunk("d1", 0, "첫 번째 청크"),
                _chunk("d1", 1, "두 번째 청크"),
                _chunk("d1", 2, "세 번째 청크"),
            ],
        )
        stats = await ingester.ingest(InMemoryDocumentSource([doc]))

        assert stats.chunks_created == 3
        # 3 CONTAINS + 2 NEXT_CHUNK = 5 edges
        assert stats.edges_created == 5

        chunks = await backend.list_nodes(kind=NodeKind.CHUNK, limit=100)
        assert len(chunks) == 3
        # Chunks sorted by index
        texts = [c.content for c in sorted(chunks, key=lambda c: int(c.properties.get("chunk_index", 0)))]
        assert texts == ["첫 번째 청크", "두 번째 청크", "세 번째 청크"]

    @pytest.mark.asyncio
    async def test_category_created_once_across_docs(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        ingester = DocumentIngester(profile=profile, backend=backend)

        docs = [
            _sample_doc("d1", "First", category="규정 및 지침"),
            _sample_doc("d2", "Second", category="규정 및 지침"),
            _sample_doc("d3", "Third", category="운영계획"),
        ]
        stats = await ingester.ingest(InMemoryDocumentSource(docs))

        assert stats.categories_created == 2
        assert stats.documents_ingested == 3

        categories = await backend.list_nodes(kind=NodeKind.CONCEPT, limit=100)
        cat_titles = {c.title for c in categories}
        assert "규정 및 지침" in cat_titles
        assert "운영계획" in cat_titles


# --- Ontology hints drive NodeKind ---


class TestOntologyHints:
    @pytest.mark.asyncio
    async def test_category_maps_to_node_kind(self):
        backend = MemoryBackend()
        profile = DomainProfile(
            name="test",
            locale="ko",
            ontology_hints={
                "규정 및 지침": NodeKind.RULE,
                "운영계획": NodeKind.DECISION,
                "조사 및 평가": NodeKind.OBSERVATION,
            },
        )
        ingester = DocumentIngester(profile=profile, backend=backend)

        docs = [
            _sample_doc("r1", "A Rule", category="규정 및 지침"),
            _sample_doc("p1", "A Plan", category="운영계획"),
            _sample_doc("e1", "An Eval", category="조사 및 평가"),
            _sample_doc("u1", "Unclassified", category="기타"),
        ]
        await ingester.ingest(InMemoryDocumentSource(docs))

        rules = await backend.list_nodes(kind=NodeKind.RULE, limit=100)
        plans = await backend.list_nodes(kind=NodeKind.DECISION, limit=100)
        evals = await backend.list_nodes(kind=NodeKind.OBSERVATION, limit=100)
        # Unclassified category falls back to ENTITY
        entities = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)

        assert any(n.title == "A Rule" for n in rules)
        assert any(n.title == "A Plan" for n in plans)
        assert any(n.title == "An Eval" for n in evals)
        assert any(n.title == "Unclassified" for n in entities)


# --- Merge strategy ---


class TestMergeStrategy:
    @pytest.mark.asyncio
    async def test_skip_on_duplicate_doc_id(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        ingester = DocumentIngester(
            profile=profile, backend=backend, merge_strategy="skip"
        )

        doc_v1 = _sample_doc("d1", "Original", content="v1 본문")
        await ingester.ingest(InMemoryDocumentSource([doc_v1]))

        doc_v2 = _sample_doc("d1", "Updated", content="v2 본문")
        stats2 = await ingester.ingest(InMemoryDocumentSource([doc_v2]))

        assert stats2.documents_ingested == 0
        assert stats2.documents_skipped == 1

        # Original title preserved
        docs = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)
        assert any(n.title == "Original" for n in docs)
        assert not any(n.title == "Updated" for n in docs)

    @pytest.mark.asyncio
    async def test_replace_overwrites_doc_and_chunks(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()

        skipper = DocumentIngester(
            profile=profile, backend=backend, merge_strategy="skip"
        )
        v1 = _sample_doc(
            "d1", "V1", chunks=[_chunk("d1", 0, "old chunk")]
        )
        await skipper.ingest(InMemoryDocumentSource([v1]))

        replacer = DocumentIngester(
            profile=profile, backend=backend, merge_strategy="replace"
        )
        v2 = _sample_doc(
            "d1", "V2", chunks=[_chunk("d1", 0, "new chunk a"), _chunk("d1", 1, "new chunk b")]
        )
        stats = await replacer.ingest(InMemoryDocumentSource([v2]))

        assert stats.documents_ingested == 1
        assert stats.chunks_created == 2

        docs = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)
        assert any(n.title == "V2" for n in docs)
        assert not any(n.title == "V1" for n in docs)

        chunks = await backend.list_nodes(kind=NodeKind.CHUNK, limit=100)
        assert len(chunks) == 2
        chunk_texts = {c.content for c in chunks}
        assert "new chunk a" in chunk_texts
        assert "new chunk b" in chunk_texts
        assert "old chunk" not in chunk_texts

    @pytest.mark.asyncio
    async def test_invalid_merge_strategy_raises(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        with pytest.raises(ValueError, match="Unknown merge_strategy"):
            DocumentIngester(
                profile=profile, backend=backend, merge_strategy="nuclear"  # type: ignore[arg-type]
            )


# --- Idempotency under skip ---


class TestIdempotent:
    @pytest.mark.asyncio
    async def test_repeated_ingest_stable_count(self):
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        ingester = DocumentIngester(profile=profile, backend=backend)

        docs = [
            _sample_doc(
                "d1",
                "Doc 1",
                category="규정 및 지침",
                chunks=[_chunk("d1", 0, "a"), _chunk("d1", 1, "b")],
            )
        ]

        stats1 = await ingester.ingest(InMemoryDocumentSource(docs))
        stats2 = await ingester.ingest(InMemoryDocumentSource(docs))
        stats3 = await ingester.ingest(InMemoryDocumentSource(docs))

        assert stats1.documents_ingested == 1
        assert stats2.documents_ingested == 0
        assert stats3.documents_ingested == 0

        nodes = await backend.list_nodes(limit=100)
        doc_count = sum(1 for n in nodes if "document" in (n.tags or []))
        chunk_count = sum(1 for n in nodes if n.kind == NodeKind.CHUNK)
        cat_count = sum(1 for n in nodes if "category" in (n.tags or []))
        assert doc_count == 1
        assert chunk_count == 2
        assert cat_count == 1


# --- JSONL source ---


class TestJsonlDocumentSource:
    def test_reads_docs_only_file(self, tmp_path):
        docs_path = tmp_path / "docs.jsonl"
        docs_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "doc_id": "a",
                            "title": "Doc A",
                            "content": "content a",
                            "category": "cat1",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {"doc_id": "b", "title": "Doc B", "category": "cat2"},
                        ensure_ascii=False,
                    ),
                ]
            ),
            encoding="utf-8",
        )

        source = JsonlDocumentSource(docs_path)
        docs = list(source.documents())
        assert len(docs) == 2
        assert docs[0].doc_id == "a"
        assert docs[0].title == "Doc A"
        assert docs[0].category == "cat1"
        assert docs[1].doc_id == "b"

    def test_reads_docs_plus_separate_chunks_file(self, tmp_path):
        docs_path = tmp_path / "docs.jsonl"
        chunks_path = tmp_path / "chunks.jsonl"

        docs_path.write_text(
            json.dumps(
                {"doc_id": "a", "title": "Doc A", "category": "cat1"},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        chunks_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "chunk_id": "a_c0000",
                            "doc_id": "a",
                            "text": "first",
                            "index": 0,
                        }
                    ),
                    json.dumps(
                        {
                            "chunk_id": "a_c0001",
                            "doc_id": "a",
                            "text": "second",
                            "index": 1,
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        source = JsonlDocumentSource(docs_path, chunks_path)
        docs = list(source.documents())
        assert len(docs) == 1
        assert len(docs[0].chunks) == 2
        assert docs[0].chunks[0].text == "first"
        assert docs[0].chunks[1].index == 1

    def test_inline_chunks_override_separate_file(self, tmp_path):
        docs_path = tmp_path / "docs.jsonl"
        chunks_path = tmp_path / "chunks.jsonl"

        docs_path.write_text(
            json.dumps(
                {
                    "doc_id": "a",
                    "title": "Doc A",
                    "chunks": [{"chunk_id": "inl", "doc_id": "a", "text": "inline", "index": 0}],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        chunks_path.write_text(
            json.dumps(
                {"chunk_id": "other", "doc_id": "a", "text": "should not win", "index": 0}
            )
            + "\n",
            encoding="utf-8",
        )

        source = JsonlDocumentSource(docs_path, chunks_path)
        docs = list(source.documents())
        assert len(docs[0].chunks) == 1
        assert docs[0].chunks[0].text == "inline"

    def test_empty_lines_skipped(self, tmp_path):
        docs_path = tmp_path / "docs.jsonl"
        docs_path.write_text(
            "\n"
            + json.dumps({"doc_id": "a", "title": "A"})
            + "\n\n"
            + json.dumps({"doc_id": "b", "title": "B"})
            + "\n\n",
            encoding="utf-8",
        )
        source = JsonlDocumentSource(docs_path)
        docs = list(source.documents())
        assert len(docs) == 2

    def test_missing_doc_id_skipped(self, tmp_path):
        docs_path = tmp_path / "docs.jsonl"
        docs_path.write_text(
            json.dumps({"title": "Missing id"}) + "\n"
            + json.dumps({"doc_id": "ok", "title": "OK"}) + "\n",
            encoding="utf-8",
        )
        source = JsonlDocumentSource(docs_path)
        docs = list(source.documents())
        assert len(docs) == 1
        assert docs[0].doc_id == "ok"


# --- Full KRRA-style integration (without KRRA hardcoding) ---


class TestKrraStyleIntegration:
    @pytest.mark.asyncio
    async def test_krra_profile_ingest_via_jsonl(self, tmp_path):
        """Simulate the KRRA pipeline: JSONL → profile → DocumentIngester."""
        docs_path = tmp_path / "docs.jsonl"
        chunks_path = tmp_path / "chunks.jsonl"

        docs_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "doc_id": "krra_d1",
                            "title": "2020년 인권영향평가 체크리스트",
                            "category": "인권경영",
                            "source_path": "인권경영/한글/...",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "doc_id": "krra_d2",
                            "title": "2020년 온실가스 감축 계획",
                            "category": "ESG 및 지속가능성",
                        },
                        ensure_ascii=False,
                    ),
                ]
            ),
            encoding="utf-8",
        )
        chunks_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "chunk_id": "krra_d1_c0000",
                            "doc_id": "krra_d1",
                            "text": "인권영향평가 도입 배경",
                            "index": 0,
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "chunk_id": "krra_d2_c0000",
                            "doc_id": "krra_d2",
                            "text": "온실가스 감축 목표 설정",
                            "index": 0,
                        },
                        ensure_ascii=False,
                    ),
                ]
            ),
            encoding="utf-8",
        )

        profile = DomainProfile(
            name="krra_like",
            locale="ko",
            ontology_hints={
                "인권경영": NodeKind.RULE,
                "ESG 및 지속가능성": NodeKind.RULE,
            },
        )
        backend = MemoryBackend()
        ingester = DocumentIngester(profile=profile, backend=backend)

        stats = await ingester.ingest(JsonlDocumentSource(docs_path, chunks_path))

        assert stats.documents_ingested == 2
        assert stats.chunks_created == 2
        assert stats.categories_created == 2

        rules = await backend.list_nodes(kind=NodeKind.RULE, limit=100)
        assert len(rules) == 2
        titles = {n.title for n in rules}
        assert "2020년 인권영향평가 체크리스트" in titles
        assert "2020년 온실가스 감축 계획" in titles

    @pytest.mark.asyncio
    async def test_doc_content_defaults_to_title_for_fts(self):
        """Docs without body content should still be FTS-findable by title."""
        backend = MemoryBackend()
        profile = DomainProfile.generic_korean()
        ingester = DocumentIngester(profile=profile, backend=backend)

        await ingester.ingest(
            InMemoryDocumentSource(
                [_sample_doc("d1", "인권영향평가 체크리스트")]
            )
        )

        docs = await backend.list_nodes(kind=NodeKind.ENTITY, limit=100)
        target = next(n for n in docs if n.title == "인권영향평가 체크리스트")
        assert target.content  # not empty
        assert "인권영향평가" in target.content
