"""SynapticGraph — main entry point (facade)."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from time import time
from typing import TYPE_CHECKING

from synaptic.agent_search import AgentSearch, SearchIntent, suggest_intent
from synaptic.cache import NodeCache
from synaptic.consolidation import ConsolidationCascade
from synaptic.evidence import EvidenceAssembler
from synaptic.exporter import JSONExporter, MarkdownExporter
from synaptic.extensions.embedder import EmbeddingProvider
from synaptic.extensions.phrase_extractor import PhraseExtractor
from synaptic.hebbian import HebbianEngine
from synaptic.models import (
    ConsolidationLevel,
    DigestResult,
    Edge,
    EdgeKind,
    EvidenceChain,
    MaintenanceResult,
    Node,
    NodeKind,
    SearchResult,
)
from synaptic.ontology import OntologyRegistry, build_agent_ontology
from synaptic.protocols import (
    Digester,
    KindClassifier,
    QueryRewriter,
    RelationDetector,
    StorageBackend,
    TagExtractor,
)
from synaptic.search import HybridSearch
from synaptic.store import Store

if TYPE_CHECKING:
    from synaptic.extensions.llm_provider import LLMProvider


class SynapticGraph:
    """Facade over the synaptic memory system.

    Quick Start::

        # 1. In-memory (zero-dep, testing/prototyping)
        graph = SynapticGraph.memory()

        # 2. SQLite (lightweight production)
        graph = SynapticGraph.sqlite("knowledge.db")

        # 3. Full preset with custom backend
        graph = SynapticGraph(backend, classifier=..., embedder=...)
    """

    __slots__ = (
        "_agent_search",
        "_backend",
        "_cache",
        "_classifier",
        "_consolidation",
        "_corpus_size",
        "_embedder",
        "_hebbian",
        "_json_exporter",
        "_md_exporter",
        "_ontology",
        "_phrase_extractor",
        "_relation_detector",
        "_search",
        "_store",
    )

    def __init__(
        self,
        backend: StorageBackend,
        *,
        query_rewriter: QueryRewriter | None = None,
        tag_extractor: TagExtractor | None = None,
        ontology: OntologyRegistry | None = None,
        embedder: EmbeddingProvider | None = None,
        classifier: KindClassifier | None = None,
        relation_detector: RelationDetector | None = None,
        phrase_extractor: PhraseExtractor | None = None,
        cache_size: int = 256,
    ) -> None:
        self._backend = backend
        self._store = Store(backend, tag_extractor=tag_extractor)
        self._search = HybridSearch(query_rewriter=query_rewriter)
        self._hebbian = HebbianEngine()
        self._consolidation = ConsolidationCascade()
        self._md_exporter = MarkdownExporter()
        self._json_exporter = JSONExporter()
        self._cache = NodeCache(maxsize=cache_size)
        self._ontology = ontology
        self._embedder = embedder
        self._classifier = classifier
        self._relation_detector = relation_detector
        self._phrase_extractor = phrase_extractor
        self._agent_search = AgentSearch(hybrid=self._search)
        self._corpus_size = 0

    # --- Factory methods ---

    @classmethod
    def memory(cls, *, cache_size: int = 256) -> SynapticGraph:
        """In-memory backend — zero dependencies, for testing/prototyping.

        Example::

            graph = SynapticGraph.memory()
            await graph.add("Hello", "World")
        """
        from synaptic.backends.memory import MemoryBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier

        return cls(
            MemoryBackend(),
            classifier=RuleBasedClassifier(),
            cache_size=cache_size,
        )

    @classmethod
    def sqlite(
        cls,
        db_path: str = "synaptic.db",
        *,
        cache_size: int = 256,
    ) -> SynapticGraph:
        """SQLite backend — lightweight production, FTS5 search support.

        Example::

            graph = SynapticGraph.sqlite("knowledge.db")
            await graph.backend.connect()
            await graph.add("Hello", "World")
        """
        from synaptic.backends.sqlite import SQLiteBackend
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        return cls(
            SQLiteBackend(db_path),
            classifier=RuleBasedClassifier(),
            relation_detector=RuleBasedRelationDetector(),
            ontology=build_agent_ontology(),
            cache_size=cache_size,
        )

    @classmethod
    def full(
        cls,
        backend: StorageBackend,
        *,
        llm: LLMProvider | None = None,
        embed_api_base: str = "",
        embed_model: str = "default",
        embed_api_key: str = "",
        cache_size: int = 512,
    ) -> SynapticGraph:
        """Full-featured setup — LLM classification, embedding, relation detection, ontology.

        Example::

            from synaptic.backends.sqlite import SQLiteBackend
            from synaptic.extensions.llm_provider import OllamaLLMProvider

            graph = SynapticGraph.full(
                SQLiteBackend("knowledge.db"),
                llm=OllamaLLMProvider(model="gemma3:4b"),
                embed_api_base="http://localhost:8080/v1",
            )
        """
        from synaptic.extensions.classifier_rules import RuleBasedClassifier
        from synaptic.extensions.relation_detector import RuleBasedRelationDetector

        classifier: KindClassifier
        relation_detector: RelationDetector
        embedder: EmbeddingProvider | None = None

        if llm is not None:
            from synaptic.extensions.classifier_hybrid import HybridClassifier
            from synaptic.extensions.classifier_llm import LLMClassifier
            from synaptic.extensions.relation_detector_llm import (
                LLMRelationDetector,
            )

            classifier = HybridClassifier(
                llm=LLMClassifier(llm, fallback=RuleBasedClassifier()),
                rule=RuleBasedClassifier(),
            )
            relation_detector = LLMRelationDetector(llm, fallback=RuleBasedRelationDetector())
        else:
            classifier = RuleBasedClassifier()
            relation_detector = RuleBasedRelationDetector()

        if embed_api_base:
            from synaptic.extensions.embedder import OpenAIEmbeddingProvider

            embedder = OpenAIEmbeddingProvider(
                api_base=embed_api_base,
                model=embed_model,
                api_key=embed_api_key,
            )

        return cls(
            backend,
            classifier=classifier,
            relation_detector=relation_detector,
            embedder=embedder,
            ontology=build_agent_ontology(),
            phrase_extractor=PhraseExtractor(),
            cache_size=cache_size,
        )

    @property
    def backend(self) -> StorageBackend:
        return self._backend

    async def _get_corpus_size(self) -> int:
        """Get corpus size for adaptive search weighting (cached)."""
        if self._corpus_size > 0:
            return self._corpus_size
        # First call: compute from backend
        if hasattr(self._backend, "_nodes"):
            self._corpus_size = len(self._backend._nodes)  # type: ignore[attr-defined]
        else:
            nodes = await self._backend.list_nodes(limit=100000)
            self._corpus_size = len(nodes)
        return self._corpus_size

    @property
    def cache(self) -> NodeCache:
        return self._cache

    @property
    def ontology(self) -> OntologyRegistry | None:
        return self._ontology

    async def add(
        self,
        title: str,
        content: str,
        *,
        kind: str | NodeKind | None = None,
        tags: list[str] | None = None,
        source: str = "",
        embedding: list[float] | None = None,
        properties: dict[str, str] | None = None,
    ) -> Node:
        # Auto-classify kind if not specified
        if kind is None:
            if self._classifier is not None:
                # LLM classifier: generate rich metadata via classify_async
                if hasattr(self._classifier, "classify_async"):
                    result = await self._classifier.classify_async(title, content)
                    kind = result.kind
                    if tags is None:
                        tags = result.tags
                    if properties is None:
                        properties = {}
                    if result.search_keywords:
                        properties["_search_keywords"] = ",".join(result.search_keywords)
                    if result.search_scenarios:
                        properties["_search_scenarios"] = "|".join(result.search_scenarios)
                    if result.summary:
                        properties["_summary"] = result.summary
                else:
                    kind = self._classifier.classify(title, content)
            else:
                kind = NodeKind.CONCEPT

        # Validate against ontology if available
        if self._ontology and properties:
            errors = self._ontology.validate_node(str(kind), properties)
            if errors:
                msg = f"Ontology validation failed: {'; '.join(errors)}"
                raise ValueError(msg)

        # Auto-embed if embedder is available and no embedding provided
        if embedding is None and self._embedder is not None:
            # Include LLM classifier-generated metadata in the embedding text
            embed_text = f"{title} {content}".strip()
            if properties:
                search_kw = properties.get("_search_keywords", "")
                summary = properties.get("_summary", "")
                if search_kw or summary:
                    embed_text = f"{title} {summary} {search_kw} {content}".strip()
            if embed_text:
                embedding = await self._embedder.embed(embed_text)

        node = await self._store.add_node(
            title,
            content,
            kind=kind,
            tags=tags,
            source=source,
            embedding=embedding,
            properties=properties,
        )
        self._cache.put(node)
        self._corpus_size += 1

        # Auto-detect relations with existing nodes
        if self._relation_detector is not None:
            self._relation_detector.index.add(node)
            relations = await self._relation_detector.detect(node, self._backend)
            for target_id, edge_kind, weight in relations:
                await self._store.add_edge(
                    node.id,
                    target_id,
                    kind=edge_kind,
                    weight=weight,
                )

        # Phrase extraction and linking (HippoRAG2 dual-node KG)
        if self._phrase_extractor is not None:
            await self._phrase_extractor.extract_and_link(
                self,
                node.id,
                title,
                content,
            )

        return node

    async def add_document(
        self,
        title: str,
        content: str,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        kind: str | NodeKind | None = None,
        tags: list[str] | None = None,
        source: str = "",
        properties: dict[str, str] | None = None,
    ) -> list[Node]:
        """긴 문서를 자동 청킹하여 여러 노드로 추가.

        chunk_size 이하 문서는 단일 노드로 추가 (add()와 동일).
        긴 문서는 문장 경계에서 분할하고 PART_OF 관계로 연결.

        Returns:
            생성된 노드 리스트 (첫 번째가 대표 노드).
        """
        # 짧은 문서는 그냥 add()
        if len(content) <= chunk_size:
            node = await self.add(
                title=title,
                content=content,
                kind=kind,
                tags=tags,
                source=source,
                properties=properties,
            )
            return [node]

        # 문장 경계에서 청킹
        chunks = self._split_into_chunks(content, chunk_size, chunk_overlap)
        nodes: list[Node] = []
        for i, chunk in enumerate(chunks):
            chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
            chunk_tags = list(tags) if tags else []
            chunk_tags.append(f"chunk:{i}")
            if len(chunks) > 1:
                chunk_tags.append(f"chunks:{len(chunks)}")

            node = await self.add(
                title=chunk_title,
                content=chunk,
                kind=kind,
                tags=chunk_tags,
                source=source,
                properties=properties,
            )
            nodes.append(node)

        # 청크 간 PART_OF 관계 연결
        if len(nodes) > 1:
            for i in range(1, len(nodes)):
                await self.link(
                    nodes[i].id,
                    nodes[0].id,
                    kind=EdgeKind.PART_OF,
                    weight=0.9,
                )

        return nodes

    @staticmethod
    def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
        """문장 경계에서 텍스트 분할."""
        import re as _re

        sentences = _re.split(r"(?<=[.!?。\n])\s+", text)

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > chunk_size and current:
                chunks.append(" ".join(current))
                # overlap: 마지막 문장들 유지
                overlap_sents: list[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                current = overlap_sents
                current_len = overlap_len

            current.append(sent)
            current_len += len(sent)

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    async def link(
        self,
        source_id: str,
        target_id: str,
        *,
        kind: EdgeKind = EdgeKind.RELATED,
        weight: float = 1.0,
    ) -> Edge:
        # Validate against ontology relation constraints if available
        if self._ontology:
            src_node = await self._backend.get_node(source_id)
            tgt_node = await self._backend.get_node(target_id)
            if src_node is not None and tgt_node is not None:
                errors = self._ontology.validate_edge(
                    str(kind),
                    str(src_node.kind),
                    str(tgt_node.kind),
                )
                if errors:
                    msg = f"Ontology validation failed: {'; '.join(errors)}"
                    raise ValueError(msg)
        return await self._store.add_edge(source_id, target_id, kind=kind, weight=weight)

    async def search(
        self,
        query: str,
        *,
        limit: int = 10,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        # Auto-embed query for vector search
        if embedding is None and self._embedder is not None:
            embedding = await self._embedder.embed(query)
        # Corpus size for adaptive vector weighting
        corpus_size = await self._get_corpus_size()
        return await self._search.search(
            self._backend, query, limit=limit, embedding=embedding, corpus_size=corpus_size
        )

    async def agent_search(
        self,
        query: str,
        *,
        intent: str = "auto",
        context_tags: list[str] | None = None,
        limit: int = 10,
        embedding: list[float] | None = None,
        depth: int = 2,
    ) -> SearchResult:
        """Agent-optimized search with intent and context awareness.

        Set intent="auto" (default) to infer intent from query keywords.
        """
        # Auto-embed query for vector search
        if embedding is None and self._embedder is not None:
            embedding = await self._embedder.embed(query)
        if intent == "auto":
            search_intent = suggest_intent(query)
        else:
            search_intent = SearchIntent(intent)
        corpus_size = await self._get_corpus_size()
        return await self._agent_search.search(
            self._backend,
            query,
            intent=search_intent,
            context_tags=context_tags,
            limit=limit,
            embedding=embedding,
            depth=depth,
            corpus_size=corpus_size,
        )

    async def list(
        self,
        *,
        kind: str | NodeKind | None = None,
        level: ConsolidationLevel | None = None,
        limit: int = 100,
    ) -> list[Node]:
        """List all nodes with optional kind/level filtering."""
        return await self._backend.list_nodes(kind=kind, level=level, limit=limit)

    async def get(self, node_id: str) -> Node | None:
        cached = self._cache.get(node_id)
        if cached is not None:
            # Still track access in backend for consolidation
            cached.access_count += 1
            cached.updated_at = time()
            await self._backend.update_node(cached)
            return cached
        node = await self._store.get_node(node_id)
        if node is not None:
            self._cache.put(node)
        return node

    async def update(
        self,
        node_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        kind: str | NodeKind | None = None,
        tags: list[str] | None = None,
        properties: dict[str, str] | None = None,
        embedding: list[float] | None = None,
    ) -> Node | None:
        """Update a node's fields by ID. Returns updated node, or None if not found."""
        node = await self._backend.get_node(node_id)
        if node is None:
            return None
        if title is not None:
            node.title = title
        if content is not None:
            node.content = content
        if kind is not None:
            node.kind = kind
        if tags is not None:
            node.tags = tags
        if properties is not None:
            node.properties = properties
        if embedding is not None:
            node.embedding = embedding
        node.updated_at = time()
        await self._backend.update_node(node)
        self._cache.invalidate(node_id)
        self._cache.put(node)
        return node

    async def remove(self, node_id: str) -> bool:
        node = await self._backend.get_node(node_id)
        if node is None:
            return False
        # Remove from relation detector index
        if self._relation_detector is not None:
            self._relation_detector.index.remove(node_id)
        await self._store.delete_node(node_id)
        self._cache.invalidate(node_id)
        self._corpus_size = max(0, self._corpus_size - 1)
        return True

    async def reinforce(self, node_ids: list[str], *, success: bool = True) -> None:
        await self._hebbian.reinforce(self._backend, node_ids, success=success)
        # Invalidate cached nodes (counts changed)
        for nid in node_ids:
            self._cache.invalidate(nid)

    async def consolidate(
        self,
        digester: Digester | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> DigestResult:
        return await self._consolidation.consolidate(self._backend, digester, context=context)

    async def prune(self) -> int:
        return await self._backend.prune_edges(weight_below=0.1)

    async def decay(self) -> int:
        self._cache.clear()  # Vitality changed globally
        return await self._backend.decay_vitality(factor=0.95)

    async def maintain(
        self,
        digester: Digester | None = None,
        *,
        context: dict[str, object] | None = None,
    ) -> MaintenanceResult:
        """Run consolidate + decay + prune in one call with a unified result."""
        consolidated = await self._consolidation.consolidate(
            self._backend,
            digester,
            context=context,
        )
        decayed = await self.decay()
        pruned = await self.prune()
        return MaintenanceResult(consolidated=consolidated, decayed=decayed, pruned=pruned)

    async def export_markdown(self, *, node_ids: list[str] | None = None) -> str:
        return await self._md_exporter.export(self._backend, node_ids=node_ids)

    async def export_json(self, *, node_ids: list[str] | None = None) -> str:
        return await self._json_exporter.export(self._backend, node_ids=node_ids)

    async def merge(
        self,
        source_id: str,
        target_id: str,
    ) -> Node | None:
        """Merge source node into target. Combines content, stats, edges.

        Source node is deleted after merge.
        Returns the updated target node, or None if either node is missing.
        """
        source = await self._backend.get_node(source_id)
        target = await self._backend.get_node(target_id)
        if source is None or target is None:
            return None

        # Merge content
        if source.content and source.content not in target.content:
            target.content = f"{target.content}\n\n{source.content}".strip()

        # Merge tags (deduplicate)
        merged_tags = list(dict.fromkeys([*target.tags, *source.tags]))
        target.tags = merged_tags

        # Merge stats
        target.access_count += source.access_count
        target.success_count += source.success_count
        target.failure_count += source.failure_count
        target.vitality = max(target.vitality, source.vitality)
        target.updated_at = time()

        # Re-point source's edges to target
        source_edges = await self._backend.get_edges(source_id)
        for edge in source_edges:
            new_src = target_id if edge.source_id == source_id else edge.source_id
            new_tgt = target_id if edge.target_id == source_id else edge.target_id
            if new_src != new_tgt:  # Avoid self-loops
                new_edge = Edge(
                    source_id=new_src,
                    target_id=new_tgt,
                    kind=edge.kind,
                    weight=edge.weight,
                )
                try:
                    await self._backend.save_edge(new_edge)
                except Exception:
                    pass  # Duplicate edge — skip

        await self._backend.update_node(target)
        await self._backend.delete_node(source_id)
        self._cache.invalidate(source_id)
        self._cache.invalidate(target_id)
        return target

    async def find_duplicates(
        self,
        *,
        threshold: float = 0.85,
        limit: int = 50,
    ) -> list[tuple[Node, Node, float]]:
        """Find potential duplicate node pairs based on title similarity.

        Returns list of (node_a, node_b, similarity_score) tuples.
        """
        nodes = await self._backend.list_nodes(limit=limit * 10)
        duplicates: list[tuple[Node, Node, float]] = []

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if nodes[i].kind != nodes[j].kind:
                    continue
                sim = SequenceMatcher(None, nodes[i].title.lower(), nodes[j].title.lower()).ratio()
                if sim >= threshold:
                    duplicates.append((nodes[i], nodes[j], sim))

        duplicates.sort(key=lambda x: x[2], reverse=True)
        return duplicates[:limit]

    async def stats(self) -> dict[str, int | float]:
        all_nodes = await self._backend.list_nodes(limit=10000)
        by_kind: dict[str, int] = {}
        by_level: dict[str, int] = {}
        for node in all_nodes:
            by_kind[str(node.kind)] = by_kind.get(str(node.kind), 0) + 1
            by_level[str(node.level)] = by_level.get(str(node.level), 0) + 1

        result: dict[str, int | float] = {"total_nodes": len(all_nodes)}
        for k, v in sorted(by_kind.items()):
            result[f"kind_{k}"] = v
        for k, v in sorted(by_level.items()):
            result[f"level_{k}"] = v

        cache_stats = self._cache.stats()
        result["cache_hit_rate"] = cache_stats["hit_rate"]
        result["cache_size"] = cache_stats["size"]
        return result

    async def build_evidence(
        self,
        query: str,
        *,
        search_result: SearchResult | None = None,
        limit: int = 10,
        max_steps: int = 8,
        max_tokens: int = 2048,
        max_sentences_per_node: int = 5,
        relevance_threshold: float = 0.2,
        embedding: list[float] | None = None,
    ) -> EvidenceChain:
        """Convert search results into an evidence chain optimized for small LLMs."""
        if search_result is None:
            if embedding is None and self._embedder is not None:
                embedding = await self._embedder.embed(query)
            search_result = await self.search(query, limit=limit, embedding=embedding)

        assembler = EvidenceAssembler(
            max_sentences_per_node=max_sentences_per_node,
            relevance_threshold=relevance_threshold,
            max_tokens=max_tokens,
        )
        return await assembler.assemble(
            self._backend,
            query,
            search_result,
            max_steps=max_steps,
        )

    # --- Conversation helpers ---

    async def add_turn(
        self,
        user_msg: str,
        assistant_msg: str,
        *,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> tuple[Node, Node, Node]:
        """Add a conversation turn (user + assistant) linked to a session.

        Creates a SESSION node on first call for a given session_id.
        Returns (session_node, user_node, assistant_node).
        """
        from synaptic.models import _new_id

        if session_id is None:
            session_id = f"session_{_new_id()}"

        # Get or create session node
        session_node = await self._backend.get_node(session_id)
        if session_node is None:
            session_node = await self._store.add_node(
                f"Session {session_id[:8]}",
                "",
                kind=NodeKind.SESSION,
                tags=["_session"],
                source=session_id,
            )
            # Override the auto-generated ID with session_id
            await self._backend.delete_node(session_node.id)
            session_node.id = session_id
            await self._backend.save_node(session_node)

        turn_tags = [*tags] if tags else []

        # Create user message node
        user_node = await self._store.add_node(
            "user",
            user_msg,
            kind=NodeKind.OBSERVATION,
            tags=[*turn_tags, "_turn_user"],
        )

        # Create assistant message node
        assistant_node = await self._store.add_node(
            "assistant",
            assistant_msg,
            kind=NodeKind.OBSERVATION,
            tags=[*turn_tags, "_turn_assistant"],
        )

        # Link: user → assistant (FOLLOWED_BY)
        await self._store.add_edge(
            user_node.id,
            assistant_node.id,
            kind=EdgeKind.FOLLOWED_BY,
        )

        # Link: session → user (CONTAINS)
        await self._store.add_edge(
            session_id,
            user_node.id,
            kind=EdgeKind.CONTAINS,
        )

        # Link last turn to this one (FOLLOWED_BY)
        session_edges = await self._backend.get_edges(session_id, direction="outgoing")
        contained = [
            e for e in session_edges if e.kind == EdgeKind.CONTAINS and e.target_id != user_node.id
        ]
        if contained:
            # Find the most recent contained user node
            last_user_id = contained[-1].target_id
            # Get the assistant node linked from last user
            last_edges = await self._backend.get_edges(last_user_id, direction="outgoing")
            last_assistant = [e for e in last_edges if e.kind == EdgeKind.FOLLOWED_BY]
            if last_assistant:
                await self._store.add_edge(
                    last_assistant[-1].target_id,
                    user_node.id,
                    kind=EdgeKind.FOLLOWED_BY,
                )

        return session_node, user_node, assistant_node

    # --- Ontology persistence ---

    async def save_ontology(self) -> None:
        """Persist the OntologyRegistry to the graph as a TYPE_DEF node."""
        if self._ontology is None:
            return
        data = self._ontology.to_dict()
        # Use a fixed ID so we can find/update it
        node = Node(
            id="_ontology_schema_",
            kind=NodeKind.TYPE_DEF,
            title="Ontology Schema",
            content=json.dumps(data),
            tags=["_ontology", "_system"],
            level=ConsolidationLevel.L3_PERMANENT,
        )
        await self._backend.save_node(node)

    async def load_ontology(self) -> OntologyRegistry | None:
        """Load OntologyRegistry from the graph. Returns None if not found."""
        node = await self._backend.get_node("_ontology_schema_")
        if node is None:
            return None
        try:
            data = json.loads(node.content)
            registry = OntologyRegistry.from_dict(data)
            self._ontology = registry
            return registry
        except (json.JSONDecodeError, KeyError):
            return None
