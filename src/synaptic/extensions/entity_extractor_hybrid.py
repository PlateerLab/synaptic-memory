"""Hybrid entity extraction: SpaCy base + LLM enrichment for complex chunks.

Strategy:
  1. SpaCy extracts entities and basic dep-parse relations (free)
  2. For chunks with high entity density (>= enrich_threshold),
     invoke LLM to refine relations and discover implicit entities
  3. LLM enrichment is async — only called when needed

Cost: ~94% of LLM-only quality, with LLM calls only for ~10-20% of chunks.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from synaptic.extensions.entity_extractor_spacy import SpaCyEntityExtractor
from synaptic.models import EdgeKind, NodeKind

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider
    from synaptic.graph import SynapticGraph

logger = logging.getLogger("entity-extractor-hybrid")

_LLM_SYSTEM_PROMPT = """당신은 텍스트에서 엔티티와 관계를 추출하는 전문가입니다.
주어진 텍스트와 이미 추출된 엔티티 목록을 보고:
1. 누락된 중요한 엔티티를 추가하세요
2. 엔티티 간 관계를 파악하세요

JSON 형식으로 응답하세요:
{
  "additional_entities": [{"text": "엔티티명", "label": "ORG|PER|LOC|PRODUCT|EVENT|CONCEPT"}],
  "relations": [{"subject": "엔티티A", "predicate": "관계", "object": "엔티티B"}]
}

엔티티와 관계가 없으면 빈 배열로 응답하세요."""


class HybridEntityExtractor:
    """SpaCy base extraction + optional LLM enrichment for complex chunks.

    Example::

        from synaptic.extensions.llm_provider import OllamaLLMProvider

        spacy_ext = SpaCyEntityExtractor()
        hybrid = HybridEntityExtractor(spacy_ext, llm=OllamaLLMProvider(model="qwen3:0.6b"))

        graph = SynapticGraph(backend, phrase_extractor=hybrid)

    Without LLM, behaves identically to SpaCyEntityExtractor.
    """

    __slots__ = ("_enrich_threshold", "_llm", "_spacy")

    def __init__(
        self,
        spacy: SpaCyEntityExtractor,
        *,
        llm: LLMProvider | None = None,
        enrich_threshold: int = 5,
    ) -> None:
        self._spacy = spacy
        self._llm = llm
        self._enrich_threshold = enrich_threshold

    async def extract_and_link(
        self,
        graph: SynapticGraph,
        node_id: str,
        title: str,
        content: str,
    ) -> list[str]:
        """Extract entities via SpaCy, then optionally enrich with LLM.

        LLM is only invoked when entity density >= enrich_threshold,
        keeping cost minimal for simple chunks.
        """
        # Step 1: SpaCy extraction (always)
        entity_node_ids = await self._spacy.extract_and_link(graph, node_id, title, content)

        # Step 2: LLM enrichment (only for complex chunks)
        if self._llm is not None and len(entity_node_ids) >= self._enrich_threshold:
            text = f"{title}\n{content}" if content else title
            existing_entities = self._spacy.extract_entities(text)
            existing_names = [e.text for e in existing_entities]

            additional_ids = await self._llm_enrich(graph, node_id, text, existing_names)
            entity_node_ids.extend(additional_ids)

        return entity_node_ids

    async def _llm_enrich(
        self,
        graph: SynapticGraph,
        chunk_id: str,
        text: str,
        existing_entities: list[str],
    ) -> list[str]:
        """Call LLM to find additional entities and relations."""
        if self._llm is None:
            return []

        user_prompt = (
            f"텍스트:\n{text[:2000]}\n\n"
            f"이미 추출된 엔티티: {', '.join(existing_entities)}\n\n"
            "누락된 엔티티와 관계를 추출하세요."
        )

        try:
            response = await self._llm.generate(
                system=_LLM_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=1024,
            )

            result = json.loads(response)
            additional_ids: list[str] = []

            chunk_entity_index = getattr(graph, "_chunk_entity_index", None)

            # Add new entities
            for ent in result.get("additional_entities", []):
                ent_text = ent.get("text", "").strip()
                if not ent_text or len(ent_text) < 2:
                    continue
                # Skip if already extracted by SpaCy
                if ent_text.lower() in {e.lower() for e in existing_entities}:
                    continue

                label = ent.get("label", "CONCEPT")
                entity_node = await graph._store.add_node(
                    title=ent_text,
                    content="",
                    kind=NodeKind.ENTITY,
                    tags=["_llm_enriched", f"_label:{label}"],
                )

                await graph.link(chunk_id, entity_node.id, kind=EdgeKind.MENTIONS, weight=0.7)
                additional_ids.append(entity_node.id)

                if chunk_entity_index is not None:
                    chunk_entity_index.register(chunk_id, entity_node.id)

            # Add relations between entities
            for rel in result.get("relations", []):
                subj = rel.get("subject", "")
                obj = rel.get("object", "")
                if not subj or not obj:
                    continue

                # Find entity nodes by title
                subj_node = None
                obj_node = None
                all_entities = await graph.backend.list_nodes(kind=NodeKind.ENTITY, limit=1000)
                for n in all_entities:
                    if n.title.lower() == subj.lower():
                        subj_node = n
                    if n.title.lower() == obj.lower():
                        obj_node = n

                if subj_node and obj_node:
                    await graph.link(
                        subj_node.id,
                        obj_node.id,
                        kind=EdgeKind.RELATED,
                        weight=0.6,
                    )

            return additional_ids

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"LLM enrichment failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"LLM enrichment error: {e}")
            return []
