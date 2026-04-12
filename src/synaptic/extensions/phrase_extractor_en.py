"""English phrase extractor — regex-based, zero-dep.

Split out of the original ``phrase_extractor.py`` in v0.12 so each locale
lives in its own module. For locale-aware dispatch use
``create_phrase_extractor`` from ``phrase_extractor``. For Korean corpora
use ``KoreanPhraseExtractor`` from ``phrase_extractor_ko``.

This module preserves the original HippoRAG2 dual-node KG behaviour for
English: it extracts proper nouns, single capitalised words, and
parenthesised abbreviations, then reuses the Korean-quoted / Korean-parens
patterns as a fallback for mixed-locale corpora.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph


# --- English phrase patterns ---

# Proper nouns: consecutive words starting with uppercase (2+ words or single uppercase word)
_RE_PROPER_NOUN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

# Single capitalized word (3+ chars)
_RE_SINGLE_PROPER = re.compile(r"\b([A-Z][a-z]{2,})\b")

# Abbreviations in parentheses: (MSU), (API), (LLM), etc.
_RE_ABBREVIATION = re.compile(r"\(([A-Z]{2,8})\)")

# Korean proper nouns (for mixed-locale fallback only) — kept here so that
# a user with ``locale="en"`` but occasional Korean quotes still gets
# something reasonable. Korean-only corpora should use
# ``KoreanPhraseExtractor`` directly for much higher recall.
_RE_KO_QUOTED = re.compile(
    "[\u300c\u300e\u201c\u2018]([\u0020-\u007e\uac00-\ud7a3\u3131-\u3163\u00b7\\-]+)[\u300d\u300f\u201d\u2019]"
)
_RE_KO_PARENS = re.compile(r"\((?:주|사|재|학|재단|사단)\)([\w]+)")


# --- English stopwords ---
# Phrases composed only of these are filtered out.
_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "where",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "there",
        "here",
        "not",
        "no",
        "nor",
        "so",
        "for",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "by",
        "with",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "about",
        "up",
        "down",
        "very",
        "just",
        "also",
        "than",
        "too",
        "only",
        "own",
        "same",
        "such",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "all",
        "any",
        "every",
        "new",
    }
)


def _normalize_phrase(phrase: str) -> str:
    """Normalize a phrase: strip + NFC normalization."""
    return unicodedata.normalize("NFC", phrase.strip())


def _is_meaningful(phrase: str) -> bool:
    """Check if a phrase is meaningful.

    Exclusion criteria:
    - Phrases composed only of stop words
    - Phrases composed only of digits
    - Single-character phrases
    """
    stripped = phrase.strip()
    if len(stripped) < 2:
        return False
    if stripped.isdigit():
        return False
    words = phrase.lower().split()
    non_stop = [w for w in words if w not in _STOP_WORDS]
    return len(non_stop) > 0


class EnglishPhraseExtractor:
    """Extract key phrases (English) from documents and add them as hub
    ENTITY nodes to the graph.

    Inspired by HippoRAG2's dual-node KG: passage nodes and phrase nodes
    are separate so that PPR can reach other passages via shared phrases
    (multi-hop bridging).

    Example::

        extractor = EnglishPhraseExtractor(max_phrases_per_node=10)
        graph = SynapticGraph(backend, phrase_extractor=extractor)
        node = await graph.add("Bonn Overview", "Bonn is a city in Germany...")

    Phrase nodes are created as ``NodeKind.ENTITY`` with the ``_phrase``
    tag so they can be distinguished from regular entity nodes.

    For Korean corpora use ``KoreanPhraseExtractor`` — this class extracts
    almost nothing from pure Korean text. For locale-agnostic dispatch use
    :func:`create_phrase_extractor` from ``phrase_extractor``.
    """

    __slots__ = ("_max_phrases", "_min_phrase_len", "_phrase_cache")

    def __init__(
        self,
        *,
        min_phrase_length: int = 2,
        max_phrases_per_node: int = 5,
    ) -> None:
        """Initialize.

        Args:
            min_phrase_length: Minimum character count for phrases.
            max_phrases_per_node: Cap on phrases extracted per passage.
        """
        self._min_phrase_len = min_phrase_length
        self._max_phrases = max_phrases_per_node
        # Normalized phrase text → node_id cache (reuses same phrase nodes).
        self._phrase_cache: dict[str, str] = {}

    def extract(self, text: str) -> set[str]:
        """Return the distinct phrases found in ``text``.

        Pure function — no graph writes. Matches the interface of
        ``KoreanPhraseExtractor.extract`` so post-processing passes like
        :class:`EntityLinker` can call either locale transparently.

        The set semantics mean a phrase occurring multiple times in the
        same passage counts once (correct for DF calculations).
        """
        phrases = self._extract_phrases(title="", content=text)
        return set(phrases)

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        """Extract phrases from a passage and link them as hub entities.

        1. Extract key phrases from title + content (regex-based, zero-dep).
        2. Add each phrase as an ENTITY type node (reuse existing by cache).
        3. Create CONTAINS edge from passage → phrase.
        4. Same phrase across passages serves as a bridge.

        Returns the list of phrase node ids linked (new or existing).
        """
        phrases = self._extract_phrases(title, content)
        if not phrases:
            return []

        phrase_node_ids: list[str] = []

        for phrase in phrases:
            normalized = _normalize_phrase(phrase).lower()

            cached_id = self._phrase_cache.get(normalized)
            if cached_id is not None:
                existing = await graph.backend.get_node(cached_id)
                if existing is not None:
                    await graph.link(
                        node_id,
                        cached_id,
                        kind=EdgeKind.CONTAINS,
                        weight=0.8,
                    )
                    phrase_node_ids.append(cached_id)
                    continue
                # Cache stale
                del self._phrase_cache[normalized]

            # Create new phrase node (use store directly to prevent
            # relation_detector recursion on phrase hub nodes).
            phrase_node = await graph._store.add_node(
                title=phrase,
                content="",
                kind=NodeKind.ENTITY,
                tags=["_phrase"],
            )
            await graph.backend.save_node(phrase_node)

            self._phrase_cache[normalized] = phrase_node.id

            await graph.link(
                node_id,
                phrase_node.id,
                kind=EdgeKind.CONTAINS,
                weight=0.8,
            )

            phrase_node_ids.append(phrase_node.id)

        return phrase_node_ids

    def _extract_phrases(self, title: str, content: str) -> list[str]:
        """Regex-based phrase extraction (English + mixed-locale fallback).

        Extraction rules:
        1. Proper nouns (consecutive capitalized words): "Lomonosov Moscow State University"
        2. Single capitalized proper nouns (3+ chars): "Bonn", "Germany"
        3. Abbreviations in parentheses: "(MSU)", "(API)"
        4. Korean proper nouns within quotes (fallback for mixed corpora)
        5. Korean (주)X patterns (fallback for mixed corpora)
        6. Title itself is always included
        """
        text = f"{title}\n{content}"
        seen: set[str] = set()
        phrases: list[str] = []

        def _add(phrase: str) -> None:
            normalized = _normalize_phrase(phrase)
            if len(normalized) < self._min_phrase_len:
                return
            key = normalized.lower()
            if key in seen:
                return
            if not _is_meaningful(normalized):
                return
            seen.add(key)
            phrases.append(normalized)

        _add(title)

        for m in _RE_PROPER_NOUN.finditer(text):
            _add(m.group(1))

        for m in _RE_SINGLE_PROPER.finditer(text):
            word = m.group(1)
            if word.lower() not in _STOP_WORDS:
                _add(word)

        for m in _RE_ABBREVIATION.finditer(text):
            _add(m.group(1))

        for m in _RE_KO_QUOTED.finditer(text):
            _add(m.group(1))
        for m in _RE_KO_PARENS.finditer(text):
            _add(m.group(1))

        return phrases[: self._max_phrases]
