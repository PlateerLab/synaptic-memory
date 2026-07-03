from __future__ import annotations

import sqlite3

import pytest

from synaptic import GraphBuildOptions, SynapticGraph, SynapticPreset
from synaptic.options import resolve_graph_build_options


def test_local_options_are_dependency_free() -> None:
    options = GraphBuildOptions.from_preset()

    assert options == GraphBuildOptions.local()
    assert options.embed_url is None
    assert options.rerank_url is None
    assert options.openie_enabled is False
    assert options.connect is False


def test_rag_preset_reads_endpoint_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNAPTIC_EMBED_URL", "http://embed.local/v1")
    monkeypatch.setenv("SYNAPTIC_EMBED_MODEL", "custom-embed")
    monkeypatch.setenv("SYNAPTIC_RERANK_URL", "http://rerank.local")
    monkeypatch.setenv("SYNAPTIC_RERANK_BACKEND", "tei")
    monkeypatch.setenv("SYNAPTIC_RERANK_MODEL", "custom-rerank")

    options = GraphBuildOptions.from_preset(SynapticPreset.RAG)

    assert options.embed_url == "http://embed.local/v1"
    assert options.embed_model == "custom-embed"
    assert options.rerank_url == "http://rerank.local"
    assert options.rerank_backend == "tei"
    assert options.rerank_model == "custom-rerank"


def test_agent_preset_enables_component_bridging() -> None:
    options = GraphBuildOptions.from_preset("agent")

    assert options.connect is True


def test_preset_values_are_stable() -> None:
    assert [preset.value for preset in SynapticPreset] == ["local", "rag", "agent", "scale"]


def test_invalid_preset_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        GraphBuildOptions.from_preset("unknown")


def test_explicit_kwargs_override_options() -> None:
    base = GraphBuildOptions(
        embed_url="http://old-embed/v1",
        rerank_url="http://old-rerank",
        openie_enabled=False,
        connect=True,
    )

    options = resolve_graph_build_options(
        options=base,
        embed_url="http://new-embed/v1",
        openie_enabled=True,
        connect=False,
    )

    assert options.embed_url == "http://new-embed/v1"
    assert options.rerank_url == "http://old-rerank"
    assert options.openie_enabled is True
    assert options.connect is False


def test_preset_and_options_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either preset"):
        resolve_graph_build_options(preset="rag", options=GraphBuildOptions.local())


async def test_from_chunks_accepts_build_options(tmp_path) -> None:
    graph = await SynapticGraph.from_chunks(
        [{"content": "A tiny chunk for options smoke testing."}],
        db=str(tmp_path / "options.db"),
        options=GraphBuildOptions.local(),
    )
    try:
        assert graph._embedder is None
        result = await graph.search("tiny chunk")
        assert result.nodes
    finally:
        await graph.close()


async def test_from_data_accepts_preset(tmp_path) -> None:
    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "name,description\n"
        "Alpha Lamp,USB-C desk lamp with warm light\n"
        "Beta Bottle,Insulated bottle for hiking\n",
        encoding="utf-8",
    )

    graph = await SynapticGraph.from_data(
        str(csv_path),
        db=str(tmp_path / "from_data.db"),
        preset="local",
    )
    try:
        result = await graph.search("USB-C desk lamp")
        assert result.nodes
    finally:
        await graph.close()


async def test_from_database_accepts_build_options(tmp_path) -> None:
    pytest.importorskip("aiosqlite")
    source_db = tmp_path / "source.db"
    with sqlite3.connect(source_db) as conn:
        conn.execute("CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, description TEXT)")
        conn.execute(
            "INSERT INTO products (id, name, description) VALUES (1, 'Alpha Lamp', 'USB-C desk lamp')"
        )

    graph = await SynapticGraph.from_database(
        f"sqlite:///{source_db}",
        db=str(tmp_path / "from_database.db"),
        options=GraphBuildOptions.local(),
    )
    try:
        result = await graph.search("Alpha Lamp")
        assert result.nodes
    finally:
        await graph.close()
