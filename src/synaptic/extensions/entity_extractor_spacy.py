"""SpaCy-based entity and relation extraction.

Drop-in replacement for PhraseExtractor — same extract_and_link() interface.
Uses ko_core_news_lg for Korean and en_core_web_sm for English NER.
Dependency-parse based relation extraction for subject-predicate-object triples.

Falls back to PhraseExtractor if SpaCy is not installed.

Requires: pip install synaptic-memory[spacy]
          python -m spacy download ko_core_news_lg
          python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.chunk_entity_index import ChunkEntityIndex
    from synaptic.graph import SynapticGraph

logger = logging.getLogger("entity-extractor-spacy")

# Korean character range detection
_RE_KOREAN = re.compile(r"[\uac00-\ud7a3]")


@dataclass(slots=True)
class ExtractedEntity:
    """A named entity extracted from text."""

    text: str
    label: str  # PER, ORG, LOC, DATE, PRODUCT, etc.
    start: int
    end: int
    confidence: float = 1.0


@dataclass(slots=True)
class ExtractedRelation:
    """A relation triple extracted via dependency parsing."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _detect_lang(text: str) -> str:
    """Simple language detection: Korean if >30% Korean chars, else English."""
    if not text:
        return "en"
    korean_chars = len(_RE_KOREAN.findall(text))
    return "ko" if korean_chars / max(len(text), 1) > 0.3 else "en"


class SpaCyEntityExtractor:
    """SpaCy NER + dependency-parse relation extraction.

    Achieves ~94% of LLM extraction performance at near-zero cost (E2GraphRAG finding).

    Example::

        extractor = SpaCyEntityExtractor()
        graph = SynapticGraph(backend, phrase_extractor=extractor)
        # Entities auto-extracted and linked on graph.add() / graph.add_document()

    Falls back to PhraseExtractor if SpaCy models are not installed.
    """

    __slots__ = (
        "_en_nlp",
        "_entity_cache",
        "_fallback",
        "_ko_nlp",
        "_max_entities",
        "_min_entity_len",
    )

    def __init__(
        self,
        *,
        ko_model: str = "ko_core_news_lg",
        en_model: str = "en_core_web_sm",
        max_entities_per_chunk: int = 15,
        min_entity_length: int = 2,
        fallback: object | None = None,
    ) -> None:
        self._max_entities = max_entities_per_chunk
        self._min_entity_len = min_entity_length
        self._fallback = fallback
        # Normalized entity text → node_id (reuse same entity nodes)
        self._entity_cache: dict[str, str] = {}

        # Lazy-load SpaCy models
        self._ko_nlp: object | None = None
        self._en_nlp: object | None = None

        try:
            import spacy

            try:
                self._ko_nlp = spacy.load(ko_model, disable=["parser", "lemmatizer"])
            except OSError:
                logger.warning(f"SpaCy model '{ko_model}' not found. Korean NER disabled.")

            try:
                self._en_nlp = spacy.load(en_model, disable=["lemmatizer"])
            except OSError:
                logger.warning(f"SpaCy model '{en_model}' not found. English NER disabled.")

        except ImportError:
            logger.warning("SpaCy not installed. Falling back to PhraseExtractor.")

    def _get_nlp(self, lang: str) -> object | None:
        return self._ko_nlp if lang == "ko" else self._en_nlp

    def extract_entities(self, text: str, *, lang: str = "auto") -> list[ExtractedEntity]:
        """Extract named entities from text using SpaCy NER."""
        if lang == "auto":
            lang = _detect_lang(text)

        nlp = self._get_nlp(lang)
        if nlp is None:
            return []

        doc = nlp(text)  # type: ignore[operator]
        entities: list[ExtractedEntity] = []
        seen: set[str] = set()

        for ent in doc.ents:  # type: ignore[attr-defined]
            normalized = _normalize(ent.text)
            key = normalized.lower()

            if len(normalized) < self._min_entity_len:
                continue
            if key in seen:
                continue
            # Skip pure numbers (dates are OK via label check)
            if normalized.isdigit() and ent.label_ not in ("DATE", "TIME", "CARDINAL"):
                continue

            seen.add(key)
            entities.append(
                ExtractedEntity(
                    text=normalized,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                )
            )

        return entities[: self._max_entities]

    def extract_relations(
        self, text: str, entities: list[ExtractedEntity], *, lang: str = "auto"
    ) -> list[ExtractedRelation]:
        """Extract relations via dependency parsing (subject-verb-object patterns)."""
        if lang == "auto":
            lang = _detect_lang(text)

        nlp = self._get_nlp(lang)
        if nlp is None:
            return []

        # Need parser for relation extraction
        try:
            import spacy

            nlp_with_parser = spacy.load("ko_core_news_lg" if lang == "ko" else "en_core_web_sm")
        except (ImportError, OSError):
            return []

        doc = nlp_with_parser(text)

        # Build entity text set for matching
        entity_texts = {e.text.lower() for e in entities}
        relations: list[ExtractedRelation] = []

        for token in doc:  # type: ignore[attr-defined]
            # Look for verb tokens with subject and object children
            if token.pos_ in ("VERB", "AUX"):
                subjects = [
                    child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")
                ]
                objects = [
                    child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")
                ]

                for subj in subjects:
                    for obj in objects:
                        subj_text = _normalize(subj.text)
                        obj_text = _normalize(obj.text)
                        # Only keep relations involving known entities
                        if subj_text.lower() in entity_texts or obj_text.lower() in entity_texts:
                            relations.append(
                                ExtractedRelation(
                                    subject=subj_text,
                                    predicate=_normalize(token.lemma_),
                                    object=obj_text,
                                    confidence=0.7,
                                )
                            )

        return relations

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        """Extract entities and link them to the chunk node.

        Drop-in replacement for PhraseExtractor.extract_and_link().

        1. SpaCy NER → extract entities
        2. Create/reuse ENTITY nodes
        3. Create MENTIONS edge (chunk → entity)
        4. Register in ChunkEntityIndex if available

        Falls back to PhraseExtractor if SpaCy is not available.
        """
        # Fallback if SpaCy not available
        if self._ko_nlp is None and self._en_nlp is None:
            if self._fallback is not None and hasattr(self._fallback, "extract_and_link"):
                return await self._fallback.extract_and_link(graph, node_id, title, content)
            return []

        text = f"{title}\n{content}" if content else title
        entities = self.extract_entities(text)

        if not entities:
            return []

        entity_node_ids: list[str] = []
        chunk_entity_index: ChunkEntityIndex | None = getattr(graph, "_chunk_entity_index", None)

        for entity in entities:
            normalized_key = entity.text.lower()

            # Check cache for existing entity node
            if normalized_key in self._entity_cache:
                entity_node_id = self._entity_cache[normalized_key]
                existing = await graph.backend.get_node(entity_node_id)
                if existing is not None:
                    # Link chunk → entity via MENTIONS
                    await graph.link(
                        node_id,
                        entity_node_id,
                        kind=EdgeKind.MENTIONS,
                        weight=0.8,
                    )
                    entity_node_ids.append(entity_node_id)
                    if chunk_entity_index is not None:
                        chunk_entity_index.register(node_id, entity_node_id)
                    continue
                del self._entity_cache[normalized_key]

            # Create new entity node (via store to avoid relation_detector duplication)
            entity_node = await graph._store.add_node(
                title=entity.text,
                content="",
                kind=NodeKind.ENTITY,
                tags=["_spacy", f"_label:{entity.label}"],
            )

            self._entity_cache[normalized_key] = entity_node.id

            # chunk → entity MENTIONS edge
            await graph.link(
                node_id,
                entity_node.id,
                kind=EdgeKind.MENTIONS,
                weight=0.8,
            )

            entity_node_ids.append(entity_node.id)
            if chunk_entity_index is not None:
                chunk_entity_index.register(node_id, entity_node.id)

        return entity_node_ids
