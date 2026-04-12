"""Phrase extractor — locale-aware dispatch facade.

Historically this module held the English regex-based ``PhraseExtractor``
class directly. In v0.12 the per-locale implementations moved out:

- English / mixed-locale fallback → ``phrase_extractor_en.py``
  (class ``EnglishPhraseExtractor``)
- Korean → ``phrase_extractor_ko.py``
  (class ``KoreanPhraseExtractor``, requires a ``DomainProfile``)

For **backward compatibility** this module still exposes
``PhraseExtractor`` as an alias for ``EnglishPhraseExtractor`` and
re-exports the helper functions ``_normalize_phrase`` and
``_is_meaningful`` along with the ``_STOP_WORDS`` set. Existing code such
as ``from synaptic.extensions.phrase_extractor import PhraseExtractor``
keeps working unchanged.

For new code, prefer :func:`create_phrase_extractor` which takes a
``DomainProfile`` and returns the appropriate per-locale implementation.

Example::

    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.extensions.phrase_extractor import create_phrase_extractor

    profile = DomainProfile.generic_korean()
    extractor = create_phrase_extractor(profile)
    graph = SynapticGraph(backend, phrase_extractor=extractor)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from synaptic.extensions.phrase_extractor_en import (
    _RE_ABBREVIATION,
    _RE_KO_PARENS,
    _RE_KO_QUOTED,
    _RE_PROPER_NOUN,
    _RE_SINGLE_PROPER,
    _STOP_WORDS,
    EnglishPhraseExtractor,
    _is_meaningful,
    _normalize_phrase,
)

if TYPE_CHECKING:
    from synaptic.extensions.domain_profile import DomainProfile
    from synaptic.extensions.phrase_extractor_ko import KoreanPhraseExtractor


# Backward-compat alias — existing `PhraseExtractor()` calls keep working.
PhraseExtractor = EnglishPhraseExtractor


__all__ = [
    "EnglishPhraseExtractor",
    "PhraseExtractor",
    "_STOP_WORDS",
    "_RE_ABBREVIATION",
    "_RE_KO_PARENS",
    "_RE_KO_QUOTED",
    "_RE_PROPER_NOUN",
    "_RE_SINGLE_PROPER",
    "_is_meaningful",
    "_normalize_phrase",
    "create_phrase_extractor",
]


def create_phrase_extractor(
    profile: DomainProfile,
    *,
    max_phrases_per_node: int = 15,
) -> EnglishPhraseExtractor | KoreanPhraseExtractor:
    """Return a locale-appropriate phrase extractor for ``profile``.

    Dispatches on ``profile.locale``:

    - ``"ko"`` → :class:`KoreanPhraseExtractor` (Korean regex + DF filter)
    - ``"en"``, ``"multi"`` (default), anything else → :class:`EnglishPhraseExtractor`

    The returned object implements the same async ``extract_and_link``
    protocol so it can be passed directly into
    ``SynapticGraph(..., phrase_extractor=...)``.

    Args:
        profile: Domain profile. Its locale decides dispatch, its
            ``stopwords_extra``, ``metadata_strip_patterns``, and
            ``min_phrase_len`` / ``max_phrase_len`` are forwarded to the
            Korean extractor. For English the ``min_phrase_len`` is used
            as ``min_phrase_length``.
        max_phrases_per_node: Cap on phrases per passage.

    Returns:
        An extractor instance ready to inject into ``SynapticGraph``.
    """
    if profile.locale == "ko":
        # Local import to avoid loading Korean-specific regex for users
        # who only ever work with English corpora.
        from synaptic.extensions.phrase_extractor_ko import KoreanPhraseExtractor

        return KoreanPhraseExtractor(
            profile=profile,
            max_phrases_per_node=max_phrases_per_node,
        )

    return EnglishPhraseExtractor(
        min_phrase_length=profile.min_phrase_len,
        max_phrases_per_node=max_phrases_per_node,
    )
