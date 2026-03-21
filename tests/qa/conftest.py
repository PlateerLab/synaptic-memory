"""QA test fixtures — real data ingestion + graph setup."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.extensions.tagger_regex import RegexTagExtractor
from synaptic.graph import SynapticGraph
from synaptic.models import NodeKind

DATA_DIR = Path(__file__).parent / "data"


def _load_json(name: str) -> list[dict[str, object]]:
    path = DATA_DIR / name
    if not path.exists():
        pytest.skip(f"Data file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    if not data:
        pytest.skip(f"Data file is empty: {path}")
    return data  # type: ignore[no-any-return]


@pytest.fixture
async def wiki_graph() -> AsyncGenerator[SynapticGraph]:
    """Graph populated with Korean Wikipedia tech articles."""
    articles = _load_json("wikipedia_ko_tech.json")

    backend = MemoryBackend()
    await backend.connect()
    tagger = RegexTagExtractor()
    graph = SynapticGraph(backend, tag_extractor=tagger)

    for article in articles:
        title = str(article.get("title", ""))
        content = str(article.get("content", ""))
        if not title or not content:
            continue
        # Truncate very long articles for performance
        if len(content) > 2000:
            content = content[:2000]
        cats = article.get("categories", [])
        tags = list(cats) if isinstance(cats, list) else []
        await graph.add(
            title=title,
            content=content,
            kind=NodeKind.CONCEPT,
            tags=[str(t) for t in tags],
            source="wikipedia:ko",
        )

    yield graph
    await backend.close()


@pytest.fixture
async def github_graph() -> AsyncGenerator[SynapticGraph]:
    """Graph populated with GitHub commits + issues."""
    backend = MemoryBackend()
    await backend.connect()
    tagger = RegexTagExtractor()
    graph = SynapticGraph(backend, tag_extractor=tagger)

    # Ingest commits
    commits = _load_json("github_commits.json")
    for commit in commits:
        msg = str(commit.get("message", ""))
        if not msg or len(msg) < 10:
            continue
        first_line = msg.split("\n", maxsplit=1)[0]
        await graph.add(
            title=first_line[:100],
            content=msg,
            kind=NodeKind.ARTIFACT,
            tags=["commit"],
            source="github:commit",
        )

    # Ingest issues
    try:
        issues = _load_json("github_issues.json")
    except Exception:
        issues = []

    for issue in issues:
        title = str(issue.get("title", ""))
        body = str(issue.get("body", "") or "")
        if not title:
            continue
        labels = issue.get("labels", [])
        tag_list = ["issue"]
        if isinstance(labels, list):
            tag_list.extend(str(lb) for lb in labels[:5])
        content = body[:1500] if body else title
        await graph.add(
            title=title[:100],
            content=content,
            kind=NodeKind.ENTITY,
            tags=tag_list,
            source="github:issue",
        )

    yield graph
    await backend.close()


@pytest.fixture
async def combined_graph() -> AsyncGenerator[SynapticGraph]:
    """Graph with both Wikipedia + GitHub data combined."""
    backend = MemoryBackend()
    await backend.connect()
    tagger = RegexTagExtractor()
    graph = SynapticGraph(backend, tag_extractor=tagger)

    # Wikipedia
    try:
        articles = _load_json("wikipedia_ko_tech.json")
        for article in articles[:50]:  # Limit for performance
            title = str(article.get("title", ""))
            content = str(article.get("content", ""))[:1500]
            if title and content:
                await graph.add(
                    title=title,
                    content=content,
                    kind=NodeKind.CONCEPT,
                    source="wikipedia:ko",
                )
    except Exception:  # noqa: S110
        pass

    # GitHub commits
    try:
        commits = _load_json("github_commits.json")
        for commit in commits[:50]:
            msg = str(commit.get("message", ""))
            if msg and len(msg) >= 10:
                await graph.add(
                    title=msg.split("\n", maxsplit=1)[0][:100],
                    content=msg,
                    kind=NodeKind.ARTIFACT,
                    source="github:commit",
                )
    except Exception:  # noqa: S110
        pass

    # GitHub issues
    try:
        issues = _load_json("github_issues.json")
        for issue in issues[:50]:
            title = str(issue.get("title", ""))
            body = str(issue.get("body", "") or "")[:1500]
            if title:
                await graph.add(
                    title=title[:100],
                    content=body or title,
                    kind=NodeKind.ENTITY,
                    source="github:issue",
                )
    except Exception:  # noqa: S110
        pass

    yield graph
    await backend.close()
