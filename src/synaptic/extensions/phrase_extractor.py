"""HippoRAG2 Dual-Node KG — automatic passage + phrase extraction and linking.

Extracts key phrases from documents and adds them as ENTITY nodes to the graph.
Separates passage nodes and phrase nodes so that PPR can reach
other passages via phrases (multi-hop bridging).

- Passage → Phrase: CONTAINS edge
- Same phrase appearing in multiple passages automatically serves as a bridge

zero-dep: regex-based phrase extraction, no LLM required.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph

# --- Phrase normalization ---

# Proper nouns: consecutive words starting with uppercase (2+ words or single uppercase word)
_RE_PROPER_NOUN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"
)

# Single capitalized word (3+ chars, excluding common English words)
_RE_SINGLE_PROPER = re.compile(
    r"\b([A-Z][a-z]{2,})\b"
)

# Abbreviations in parentheses: (MSU), (API), (LLM), etc.
_RE_ABBREVIATION = re.compile(
    r"\(([A-Z]{2,8})\)"
)

# Korean proper nouns: text within quotation marks/brackets
_RE_KO_QUOTED = re.compile(
    "[\u300c\u300e\u201c\u2018]([\u0020-\u007e\uac00-\ud7a3\u3131-\u3163\u00b7\\-]+)[\u300d\u300f\u201d\u2019]"
)

# Korean proper nouns in parentheses: (주)플래티어, (재)한국재단, etc.
_RE_KO_PARENS = re.compile(
    r"\((?:주|사|재|학|재단|사단)\)([\w]+)"
)

# Common English stop words (phrases containing only these are not recognized as phrases)
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "if", "then", "else", "when",
    "where", "how", "what", "which", "who", "whom", "whose", "there",
    "here", "not", "no", "nor", "so", "for", "of", "in", "on", "at",
    "to", "from", "by", "with", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "about", "up", "down", "very", "just",
    "also", "than", "too", "only", "own", "same", "such", "both", "each",
    "few", "more", "most", "other", "some", "all", "any", "every", "new",
})


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
    # Digits only
    if stripped.isdigit():
        return False
    words = phrase.lower().split()
    non_stop = [w for w in words if w not in _STOP_WORDS]
    return len(non_stop) > 0


class PhraseExtractor:
    """Extract key phrases from documents and add them as phrase nodes to the graph.

    Inspired by HippoRAG2's dual-node KG.
    Separates passage nodes and phrase nodes so that PPR can reach
    other passages via phrases (multi-hop bridging).

    Example::

        extractor = PhraseExtractor(max_phrases_per_node=10)
        graph = SynapticGraph(backend, phrase_extractor=extractor)
        # Phrases are automatically extracted and linked on graph.add()
        node = await graph.add("Bonn Overview", "Bonn is a city in Germany...")

    Phrase nodes are created as ``NodeKind.ENTITY`` type with
    ``_phrase`` tag automatically assigned to distinguish them from regular nodes.
    """

    __slots__ = ("_min_phrase_len", "_max_phrases", "_phrase_cache")

    def __init__(
        self,
        *,
        min_phrase_length: int = 2,
        max_phrases_per_node: int = 5,
    ) -> None:
        """Initialize PhraseExtractor.

        Args:
            min_phrase_length: Minimum character count for phrases (shorter ones are ignored).
            max_phrases_per_node: Maximum number of phrases to extract per document.
        """
        self._min_phrase_len = min_phrase_length
        self._max_phrases = max_phrases_per_node
        # Normalized phrase text → node_id cache (reuses same phrase nodes)
        self._phrase_cache: dict[str, str] = {}

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        """Extract phrases from a passage node and add them as ENTITY nodes with links.

        1. Extract key phrases from title + content (regex-based, zero-dep)
        2. Add each phrase as an ENTITY type node (reuse existing node if available)
        3. Create CONTAINS edge from passage node → phrase node
        4. If a phrase exists in other passages, it automatically serves as a bridge

        Args:
            graph: SynapticGraph instance (for adding nodes/edges).
            node_id: Passage node ID.
            title: Passage title.
            content: Passage body text.

        Returns:
            List of created phrase node IDs.
        """
        phrases = self._extract_phrases(title, content)
        if not phrases:
            return []

        phrase_node_ids: list[str] = []

        for phrase in phrases:
            normalized = _normalize_phrase(phrase).lower()

            # Look up existing phrase node ID from cache
            if normalized in self._phrase_cache:
                phrase_node_id = self._phrase_cache[normalized]
                # Verify the node actually exists
                existing = await graph.backend.get_node(phrase_node_id)
                if existing is not None:
                    # Just add CONTAINS edge to existing phrase node
                    await graph.link(
                        node_id, phrase_node_id,
                        kind=EdgeKind.CONTAINS,
                        weight=0.8,
                    )
                    phrase_node_ids.append(phrase_node_id)
                    continue
                # Cache stale → remove and create new
                del self._phrase_cache[normalized]

            # Create new phrase node (use store directly instead of graph.add
            # to prevent relation_detector duplication)
            phrase_node = await graph._store.add_node(
                title=phrase,
                content="",  # minimal content to avoid FTS noise
                kind=NodeKind.ENTITY,
                tags=["_phrase"],
            )
            await graph.backend.save_node(phrase_node)

            self._phrase_cache[normalized] = phrase_node.id

            # passage → phrase CONTAINS edge
            await graph.link(
                node_id, phrase_node.id,
                kind=EdgeKind.CONTAINS,
                weight=0.8,
            )

            phrase_node_ids.append(phrase_node.id)

        return phrase_node_ids

    def _extract_phrases(self, title: str, content: str) -> list[str]:
        """Regex-based phrase extraction.

        Extraction rules:
        1. Proper nouns (consecutive capitalized words): "Lomonosov Moscow State University"
        2. Single capitalized proper nouns (3+ chars): "Bonn", "Germany"
        3. Abbreviations in parentheses: "(MSU)", "(API)"
        4. Korean proper nouns (within quotes/brackets)
        5. Years: "1755", "2024"
        6. Title itself is included as a phrase

        Deduplicated, normalized (strip), returns up to max_phrases_per_node.

        Args:
            title: Document title.
            content: Document body text.

        Returns:
            List of extracted phrases (normalized, deduplicated).
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

        # Include title itself as a phrase
        _add(title)

        # 1. Proper nouns (consecutive capitalized words)
        for m in _RE_PROPER_NOUN.finditer(text):
            _add(m.group(1))

        # 2. Single capitalized proper nouns
        for m in _RE_SINGLE_PROPER.finditer(text):
            word = m.group(1)
            # Exclude common words at sentence start (simple heuristic)
            if word.lower() not in _STOP_WORDS:
                _add(word)

        # 3. Abbreviations in parentheses
        for m in _RE_ABBREVIATION.finditer(text):
            _add(m.group(1))

        # 4. Korean proper nouns
        for m in _RE_KO_QUOTED.finditer(text):
            _add(m.group(1))
        for m in _RE_KO_PARENS.finditer(text):
            _add(m.group(1))

        return phrases[: self._max_phrases]
