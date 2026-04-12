"""Korean phrase extractor — zero-dep regex-based.

Generic for any Korean corpus. **All** domain-specific configuration
(stopwords, metadata strip patterns, DF thresholds, ontology hints) is
injected via a ``DomainProfile``. This module contains no corpus-specific
constants.

Algorithm:
    1. NFC-normalize text.
    2. Strip domain-provided metadata patterns.
    3. Extract compound Hangul nouns (length-bounded) and bigram phrases.
    4. Apply narrow particle stripping (의/을/를/에서/부터/까지/으로)
       only when the stem remains ``min_phrase_len`` or longer — this
       preserves legitimate compounds like 회계연도 and 진단결과.
    5. Filter against locale + domain stopwords.
    6. Run any domain-provided ``entity_hint_patterns`` and include
       captured text as additional phrases.

The class implements the same ``extract_and_link`` protocol as
``extensions/phrase_extractor.py`` so it can be injected into
``SynapticGraph`` via the same slot. It does not inherit from that class
to avoid mixing English-only and Korean-only logic in one hierarchy.

Typical use::

    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.extensions.phrase_extractor_ko import KoreanPhraseExtractor

    profile = DomainProfile.generic_korean()
    extractor = KoreanPhraseExtractor(profile=profile)
    graph = SynapticGraph(backend, phrase_extractor=extractor)
    await graph.add("제목", "본문...")
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from synaptic.extensions.domain_profile import DomainProfile
from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.graph import SynapticGraph


# --- Generic Korean patterns (not domain-specific) ---

# Any contiguous Hangul sequence of 2+ chars. The length bounds are
# applied later via DomainProfile so the regex itself stays generic.
_HANGUL_RUN = re.compile(r"[가-힣]{2,}")

# Bigram pattern: two whitespace-separated Korean tokens of 2+ chars each
_HANGUL_BIGRAM = re.compile(r"([가-힣]{2,})\s+([가-힣]{2,})")

# Korean particle suffixes. Order matters for regex alternation:
# longer suffixes come first (에서 before 에) so the engine matches the
# longer form when both could apply.
#
# Deliberately EXCLUDED:
#   도 — preserves 회계연도, 사업장도 etc.
#   과 — preserves 결과, 성과, 효과, 진단결과 (very common ending)
#   만 — meaning "only"; usage patterns make it ambiguous
# All other particles are safe because either (a) the stripped stem
# falls below min_stem_len=2 and bails out, or (b) no common Korean
# noun ends with them.
_PARTICLE_SUFFIX = re.compile(r"(에서|부터|까지|으로|의|을|를|에|은|는|이|가|와)$")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text) if text else text


def _strip_particle(word: str, *, min_stem_len: int = 2) -> str:
    """Strip a trailing particle if the remaining stem is long enough.

    Default ``min_stem_len`` is 2 so short nouns like 계획 (2 chars) can
    still shed their 의/을/를 suffix. The downstream ``min_phrase_len``
    filter in ``KoreanPhraseExtractor.extract`` is responsible for
    dropping stems that end up too short to be useful hubs.
    """
    m = _PARTICLE_SUFFIX.search(word)
    if m and len(word) - len(m.group()) >= min_stem_len:
        return word[: -len(m.group())]
    return word


class KoreanPhraseExtractor:
    """Extract Korean noun phrases and link them as hub entities.

    Args:
        profile: ``DomainProfile`` providing stopwords, metadata patterns,
            length bounds, and entity hint patterns. The extractor adds
            no domain knowledge of its own.
        max_phrases_per_node: Cap on phrases extracted from a single
            passage (controls MENTIONS edge explosion).
    """

    __slots__ = ("_profile", "_max_phrases", "_phrase_cache")

    def __init__(
        self,
        *,
        profile: DomainProfile,
        max_phrases_per_node: int = 15,
    ) -> None:
        if profile.locale not in ("ko", "multi"):
            msg = (
                f"KoreanPhraseExtractor requires locale 'ko' or 'multi', "
                f"got '{profile.locale}'"
            )
            raise ValueError(msg)
        self._profile = profile
        self._max_phrases = max_phrases_per_node
        # normalized phrase -> node id, for dedup on repeated link calls
        self._phrase_cache: dict[str, str] = {}

    @property
    def profile(self) -> DomainProfile:
        return self._profile

    # --- Public extraction (pure, no graph writes) ---

    def extract(self, text: str) -> set[str]:
        """Return the distinct phrases found in ``text``.

        Pure function — no graph writes, safe to call from DF passes.
        Uses a set so the same phrase occurring multiple times in the
        same passage counts once (DF semantics).
        """
        cleaned = self._clean(text)
        phrases: set[str] = set()
        stops = self._profile.stopwords()
        min_len = self._profile.min_phrase_len
        max_len = self._profile.max_phrase_len

        # 1. Single compound nouns
        for m in _HANGUL_RUN.findall(cleaned):
            stem = _strip_particle(m)
            if not (min_len <= len(stem) <= max_len):
                continue
            if stem in stops:
                continue
            phrases.add(stem)

        # 2. Bigram compounds (noun + noun)
        for m in _HANGUL_BIGRAM.finditer(cleaned):
            w1 = _strip_particle(m.group(1), min_stem_len=2)
            w2 = _strip_particle(m.group(2), min_stem_len=2)
            if len(w1) < 2 or len(w2) < 2:
                continue
            if w1 in stops or w2 in stops:
                continue
            bigram = f"{w1} {w2}"
            if len(bigram) < min_len or len(bigram) > max_len + 5:
                continue
            phrases.add(bigram)

        # 3. Domain-provided entity hint patterns (optional)
        for pattern in self._profile.entity_hint_patterns:
            for m in pattern.finditer(cleaned):
                # Use first non-empty capture group, or the full match
                groups = [g for g in m.groups() if g]
                candidate = groups[-1] if groups else m.group(0)
                candidate = _strip_particle(_nfc(candidate).strip())
                if not candidate or candidate in stops:
                    continue
                if not (min_len <= len(candidate) <= max_len):
                    continue
                phrases.add(candidate)

        return phrases

    def _clean(self, text: str) -> str:
        """NFC + strip all domain-provided metadata patterns."""
        text = _nfc(text)
        for pattern in self._profile.metadata_strip_patterns:
            text = pattern.sub("", text)
        return text

    # --- Graph integration (compatible with PhraseExtractor protocol) ---

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        """Extract phrases from a passage and link them as hub entities.

        Matches the shape of ``extensions.phrase_extractor.PhraseExtractor.
        extract_and_link`` so ``SynapticGraph`` can accept either as the
        ``phrase_extractor=`` argument.

        Returns the list of phrase node ids linked (existing or newly
        created).
        """
        combined = f"{title}\n{content}"
        phrases = self.extract(combined)
        if not phrases:
            return []

        # Prefer longer / more specific phrases first
        ordered = sorted(phrases, key=lambda p: (-len(p), p))[: self._max_phrases]

        phrase_node_ids: list[str] = []
        for phrase in ordered:
            key = phrase.lower()

            cached_id = self._phrase_cache.get(key)
            if cached_id is not None:
                existing = await graph.backend.get_node(cached_id)
                if existing is not None:
                    await graph.link(
                        node_id,
                        cached_id,
                        kind=EdgeKind.MENTIONS,
                        weight=0.8,
                    )
                    phrase_node_ids.append(cached_id)
                    continue
                # Cache stale — fall through to recreate
                del self._phrase_cache[key]

            # Create a new hub node. Use store.add_node so
            # auto-classification / relation detection / recursive
            # phrase extraction do not fire again for the hub itself.
            phrase_node = await graph._store.add_node(
                title=phrase,
                content="",
                kind=NodeKind.ENTITY,
                tags=["_phrase"],
            )
            await graph.backend.save_node(phrase_node)

            self._phrase_cache[key] = phrase_node.id
            await graph.link(
                node_id,
                phrase_node.id,
                kind=EdgeKind.MENTIONS,
                weight=0.8,
            )
            phrase_node_ids.append(phrase_node.id)

        return phrase_node_ids
