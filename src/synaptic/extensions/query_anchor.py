"""QueryAnchorExtractor — turn a raw query into structured entry points.

In the 3rd-generation GraphRAG pipeline the indexing side is kept cheap
(relation-free, lightweight extraction) and the quality burden shifts
to the **retrieval** side: before you walk the graph you have to know
where to start. That starting point is called the *query anchor*.

A good anchor has three ingredients:

1. **Categories** — which top-level topics in the index the query
   touches. Matching categories gives the reranker a structural prior:
   a chunk inside a matched category gets a small boost even if its
   raw FTS score is slightly lower.
2. **Entities / phrases** — specific named things the query mentions
   (organisations, policies, products). These drive targeted expansion:
   a chunk containing the same entity is worth more than a chunk that
   merely shares surface words.
3. **Keywords** — the fallback token set that feeds the base FTS call.
   Always produced, even when the other two are empty.

This module does all three with **zero LLM calls**. Categories come
from a simple substring match against the corpus's
``NodeKind.CONCEPT`` nodes (the Category tier built by
``DocumentIngester``). Entities/phrases come from an injected
``PhraseExtractor`` — the same one the ingestion pipeline already
uses. Keywords are produced by a light regex tokenizer.

The extractor is deliberately backend-agnostic. It talks to any
``StorageBackend`` via ``list_nodes`` so Memory, SQLite, Kuzu, and
Composite backends all work without custom paths.

Example::

    from synaptic.backends.sqlite_graph import SqliteGraphBackend
    from synaptic.extensions.query_anchor import QueryAnchorExtractor
    from synaptic.extensions.phrase_extractor import create_phrase_extractor
    from synaptic.extensions.domain_profile import DomainProfile

    backend = SqliteGraphBackend("graph.db")
    await backend.connect()

    profile = DomainProfile.load("profiles/krra.toml")
    extractor = QueryAnchorExtractor(
        backend=backend,
        phrase_extractor=create_phrase_extractor(profile),
    )

    anchors = await extractor.extract("경마 운영계획 인권경영 지침")
    # anchors.categories == ["규정 및 지침", "운영계획"]
    # anchors.entities   == ["경마", "운영계획", "인권경영", "지침"]
    # anchors.keywords   == ["경마", "운영계획", "인권경영", "지침"]
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from synaptic.models import NodeKind

if TYPE_CHECKING:
    from synaptic.protocols import StorageBackend

logger = logging.getLogger("query-anchor")


class PhraseExtractorProtocol(Protocol):
    """Minimal contract the anchor extractor expects from a phrase extractor.

    Matches what ``EntityLinker`` already uses — both
    ``EnglishPhraseExtractor`` and ``KoreanPhraseExtractor`` implement
    this, so injection is a drop-in.
    """

    def extract(self, text: str) -> set[str]:
        """Return the distinct phrases found in ``text``."""
        ...


# --- Simple keyword tokeniser ---
#
# Bigger than a whitespace split, smaller than a full NLP pipeline.
# Keeps runs of Hangul or Latin characters that are at least 2 chars
# long; everything else is treated as a separator. This gives us
# reasonable tokens for query anchor matching without pulling in a
# tokenizer model.

_KEYWORD_TOKEN = re.compile(r"[A-Za-z가-힣]{2,}")


def _nfc(s: str) -> str:
    """NFC normalise — macOS filesystem sources often arrive as NFD."""
    return unicodedata.normalize("NFC", s) if s else s


@dataclass(slots=True)
class QueryAnchors:
    """Structured entry points for a single query.

    Attributes:
        query: The original (NFC-normalised) query string.
        keywords: Token-level seeds for FTS. Always non-empty unless
            the query was blank.
        entities: Phrases the extractor treated as content-bearing.
            Subset of ``keywords`` in most cases, but may include
            multi-word phrases the keyword splitter missed.
        categories: Category labels from the corpus that overlap with
            the query. Free text — the caller looks them up in its
            own ``ontology_hints`` or category index.
        category_node_ids: Node IDs of the matched category nodes.
            Ready to feed directly into ``backend.get_neighbors`` for
            expansion.
    """

    query: str
    keywords: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    category_node_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        """``True`` if the extractor produced nothing usable."""
        return not (self.keywords or self.entities or self.categories)


class QueryAnchorExtractor:
    """Turn a raw query string into a ``QueryAnchors`` object.

    The extractor caches the category list on first use so repeated
    queries against the same backend don't re-scan the CONCEPT nodes.
    Call ``invalidate_cache()`` after bulk ingestion if new categories
    were added.

    Args:
        backend: Storage backend to source categories from.
        phrase_extractor: Optional phrase extractor for entity-grade
            candidates. When omitted, only keyword tokenisation runs —
            entities will equal keywords.
        category_cache_limit: Max number of CONCEPT nodes to pull for
            category matching. 500 is far above realistic corpus
            taxonomies and is only a safety fuse.
        min_keyword_length: Drop keywords shorter than this many
            characters. Default 2 matches the regex.
    """

    __slots__ = (
        "_backend",
        "_category_cache",
        "_category_cache_limit",
        "_extractor",
        "_min_keyword_length",
    )

    def __init__(
        self,
        *,
        backend: StorageBackend,
        phrase_extractor: PhraseExtractorProtocol | None = None,
        category_cache_limit: int = 500,
        min_keyword_length: int = 2,
    ) -> None:
        self._backend = backend
        self._extractor = phrase_extractor
        self._category_cache: list[tuple[str, str]] | None = None
        self._category_cache_limit = category_cache_limit
        self._min_keyword_length = min_keyword_length

    async def extract(self, query: str) -> QueryAnchors:
        """Build the anchor set for ``query``.

        Returns an empty ``QueryAnchors`` (not ``None``) for blank
        input so callers don't need to nullcheck every branch.
        """
        query = _nfc(query or "").strip()
        if not query:
            return QueryAnchors(query="")

        keywords = self._extract_keywords(query)
        entities = self._extract_entities(query)
        cat_labels, cat_ids = await self._match_categories(query, keywords)

        anchors = QueryAnchors(
            query=query,
            keywords=keywords,
            entities=entities,
            categories=cat_labels,
            category_node_ids=cat_ids,
        )
        logger.debug(
            "query-anchor[%r]: %d keywords, %d entities, %d categories",
            query, len(keywords), len(entities), len(cat_labels),
        )
        return anchors

    def invalidate_cache(self) -> None:
        """Drop the cached category list.

        Call this after ingesting new documents that create new
        categories — the next ``extract`` call will reload them.
        """
        self._category_cache = None

    # --- internals ---

    def _extract_keywords(self, query: str) -> list[str]:
        """Lightweight tokeniser — used for the FTS seed set.

        Preserves order of first appearance. Lowercases Latin tokens,
        leaves Hangul unchanged (case has no meaning in Korean).
        """
        seen: set[str] = set()
        out: list[str] = []
        for tok in _KEYWORD_TOKEN.findall(query):
            if len(tok) < self._min_keyword_length:
                continue
            key = tok.lower() if tok[0].isascii() else tok
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _extract_entities(self, query: str) -> list[str]:
        """Run the injected phrase extractor, fall back to keywords.

        Phrase extractors often return multi-word strings (e.g.
        ``"인권 경영"``); we preserve those as entities so downstream
        matching can prefer multi-token anchors over bare keywords.
        """
        if self._extractor is None:
            return list(self._extract_keywords(query))
        try:
            phrases = self._extractor.extract(query)
        except Exception as exc:
            logger.warning("query-anchor: phrase extraction failed — %s", exc)
            return list(self._extract_keywords(query))

        # Stable order — longer phrases first, then alphabetical
        sorted_phrases = sorted(phrases, key=lambda p: (-len(p), p))
        return sorted_phrases

    async def _match_categories(
        self,
        query: str,
        keywords: list[str],
    ) -> tuple[list[str], list[str]]:
        """Find category nodes whose title/content intersects the query.

        Two-pass match:

        1. **Full substring**: the entire category label appears
           anywhere in the query (``"규정 및 지침"`` ⊂ query).
        2. **Keyword overlap**: at least one keyword from the query
           appears in the category title (``"경마"`` ⊂ ``"경마산업관리"``).

        Pass 1 is authoritative — a full substring match outranks a
        keyword overlap. The result keeps first-match order so the
        caller can use it as a stable category priority list.
        """
        categories = await self._load_categories()
        if not categories:
            return [], []

        q_lower = query.lower()
        keyword_set = {k.lower() for k in keywords}

        matched: list[tuple[str, str]] = []  # (label, node_id)
        seen_ids: set[str] = set()

        # Pass 1: full substring matches
        for label, node_id in categories:
            label_lower = label.lower()
            if label_lower and label_lower in q_lower and node_id not in seen_ids:
                matched.append((label, node_id))
                seen_ids.add(node_id)

        # Pass 2: keyword overlap with category label
        if keyword_set:
            for label, node_id in categories:
                if node_id in seen_ids:
                    continue
                label_lower = label.lower()
                if not label_lower:
                    continue
                if any(kw in label_lower for kw in keyword_set):
                    matched.append((label, node_id))
                    seen_ids.add(node_id)

        labels = [m[0] for m in matched]
        ids = [m[1] for m in matched]
        return labels, ids

    async def _load_categories(self) -> list[tuple[str, str]]:
        """Load and cache ``(title, id)`` pairs for CONCEPT nodes.

        Only the first call is expensive; subsequent calls reuse the
        cache until ``invalidate_cache`` is called. Failures (unknown
        backend, empty store) return an empty list so the caller can
        continue with keyword-only anchors.
        """
        if self._category_cache is not None:
            return self._category_cache

        try:
            nodes = await self._backend.list_nodes(
                kind=NodeKind.CONCEPT,
                limit=self._category_cache_limit,
            )
        except Exception as exc:
            logger.warning("query-anchor: failed to load categories — %s", exc)
            self._category_cache = []
            return []

        pairs = [
            (_nfc(n.title or ""), n.id)
            for n in nodes
            if n.title and "category" in (n.tags or [])
        ]
        self._category_cache = pairs
        logger.debug("query-anchor: cached %d category nodes", len(pairs))
        return pairs
