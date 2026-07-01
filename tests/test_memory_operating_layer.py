"""Tests for the memory operating layer."""

from __future__ import annotations

import json
from time import time
from types import SimpleNamespace

import pytest

from synaptic.backends.memory import MemoryBackend
from synaptic.backends.sqlite import SQLiteBackend
from synaptic.graph import SynapticGraph
from synaptic.models import (
    ActivatedNode,
    ConsolidationLevel,
    Edge,
    EdgeKind,
    FeedbackSignal,
    MemoryEvent,
    MemoryEventKind,
    MemoryScope,
    MemoryScore,
    MemorySignalKind,
    Node,
    NodeKind,
    RetrievalEvent,
    SearchResult,
)


async def _assert_ledger_filter_semantics(backend) -> None:
    scope_a = MemoryScope(workspace_id="ws", user_id="u1")
    scope_b = MemoryScope(workspace_id="ws", user_id="u2")
    await backend.save_memory_events_batch(
        [
            MemoryEvent(
                id="mem_old",
                kind=MemoryEventKind.INGEST,
                scope=scope_a,
                source="unit",
                source_id="old",
                created_at=10.0,
            ),
            MemoryEvent(
                id="mem_mid",
                kind=MemoryEventKind.RETRIEVAL,
                scope=scope_a,
                source="unit",
                source_id="mid",
                created_at=20.0,
            ),
            MemoryEvent(
                id="mem_new",
                kind=MemoryEventKind.UPDATE,
                scope=scope_b,
                source="unit",
                source_id="new",
                created_at=30.0,
            ),
        ]
    )
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_old",
            query="old",
            scope=scope_a,
            returned_node_ids=["old"],
            created_at=10.0,
        )
    )
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_mid",
            query="mid",
            scope=scope_a,
            returned_node_ids=["mid"],
            created_at=20.0,
        )
    )
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_new",
            query="new",
            scope=scope_b,
            returned_node_ids=["new"],
            created_at=30.0,
        )
    )

    assert [event.id for event in await backend.list_memory_events(limit=2)] == [
        "mem_new",
        "mem_mid",
    ]
    assert [
        event.id
        for event in await backend.list_memory_events(kind=MemoryEventKind.INGEST, limit=10)
    ] == ["mem_old"]
    assert [event.id for event in await backend.list_memory_events(scope=scope_a, limit=10)] == [
        "mem_mid",
        "mem_old",
    ]
    assert [event.id for event in await backend.list_memory_events(since=20.0, limit=10)] == [
        "mem_new",
        "mem_mid",
    ]
    assert [
        event.id
        for event in await backend.list_memory_events(
            kind=MemoryEventKind.RETRIEVAL,
            scope=scope_a,
            since=20.0,
            limit=10,
        )
    ] == ["mem_mid"]

    fetched = await backend.get_retrieval_event("ret_mid")
    assert fetched is not None
    assert fetched.query == "mid"
    assert [event.id for event in await backend.list_retrieval_events(limit=2)] == [
        "ret_new",
        "ret_mid",
    ]
    assert [event.id for event in await backend.list_retrieval_events(scope=scope_a, limit=10)] == [
        "ret_mid",
        "ret_old",
    ]
    assert [event.id for event in await backend.list_retrieval_events(since=20.0, limit=10)] == [
        "ret_new",
        "ret_mid",
    ]
    assert [
        event.id
        for event in await backend.list_retrieval_events(scope=scope_b, since=20.0, limit=10)
    ] == ["ret_new"]


async def _assert_memory_score_node_edge_semantics(backend) -> None:
    scope = MemoryScope(workspace_id="ws", user_id="u1")
    other_scope = MemoryScope(workspace_id="ws", user_id="u2")
    node_score = MemoryScore(
        scope_key=scope.key,
        node_id="node_a",
        access_count=3,
        success_count=2,
        score=0.7,
        updated_at=20.0,
    )
    edge_score = MemoryScore(
        scope_key=scope.key,
        edge_id="edge_ab",
        access_count=4,
        failure_count=1,
        score=0.9,
        updated_at=30.0,
    )
    other_score = MemoryScore(
        scope_key=other_scope.key,
        node_id="node_b",
        access_count=1,
        score=1.0,
        updated_at=40.0,
    )
    await backend.save_memory_score(node_score)
    await backend.save_memory_score(edge_score)
    await backend.save_memory_score(other_score)

    fetched_node = await backend.get_memory_score(scope.key, node_id="node_a")
    assert fetched_node is not None
    assert fetched_node.node_id == "node_a"
    assert fetched_node.edge_id == ""
    assert fetched_node.success_count == 2
    assert fetched_node.score == pytest.approx(0.7)

    fetched_edge = await backend.get_memory_score(scope.key, edge_id="edge_ab")
    assert fetched_edge is not None
    assert fetched_edge.node_id == ""
    assert fetched_edge.edge_id == "edge_ab"
    assert fetched_edge.failure_count == 1
    assert fetched_edge.score == pytest.approx(0.9)

    assert await backend.get_memory_score(scope.key, node_id="node_a", edge_id="edge_ab") is None
    assert await backend.get_memory_score(other_scope.key, node_id="node_a") is None

    scope_scores = await backend.list_memory_scores(scope_key=scope.key, limit=10)
    assert [score.edge_id or score.node_id for score in scope_scores] == ["edge_ab", "node_a"]

    node_filtered = await backend.list_memory_scores(
        scope_key=scope.key,
        node_ids=["node_a"],
        limit=10,
    )
    assert [score.node_id for score in node_filtered] == ["node_a"]

    edge_filtered = await backend.list_memory_scores(
        scope_key=scope.key,
        edge_ids=["edge_ab"],
        limit=10,
    )
    assert [score.edge_id for score in edge_filtered] == ["edge_ab"]

    mixed_filtered = await backend.list_memory_scores(
        scope_key=scope.key,
        node_ids=["node_a"],
        edge_ids=["edge_ab"],
        limit=10,
    )
    assert mixed_filtered == []

    limited = await backend.list_memory_scores(limit=2)
    assert [score.scope_key for score in limited] == [other_scope.key, scope.key]


@pytest.mark.asyncio
async def test_memory_backend_ledger_filters_order_scope_since_and_limit():
    backend = MemoryBackend()
    await backend.connect()
    try:
        await _assert_ledger_filter_semantics(backend)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_ledger_filters_order_scope_since_and_limit():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    try:
        await _assert_ledger_filter_semantics(backend)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_memory_backend_memory_score_node_edge_roundtrip_and_filters():
    backend = MemoryBackend()
    await backend.connect()
    try:
        await _assert_memory_score_node_edge_semantics(backend)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_memory_score_node_edge_roundtrip_and_filters():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    try:
        await _assert_memory_score_node_edge_semantics(backend)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_sqlite_memory_ledgers_scores_and_edge_properties_roundtrip():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    try:
        n1 = Node(id="a", title="A")
        n2 = Node(id="b", title="B")
        await backend.save_node(n1)
        await backend.save_node(n2)
        await backend.save_edge(
            Edge(
                id="edge_ab",
                source_id="a",
                target_id="b",
                kind=EdgeKind.RELATED,
                properties={"source_event_id": "evt_1", "confidence": "0.9"},
            )
        )

        edges = await backend.get_edges("a", direction="outgoing")
        assert edges[0].properties["source_event_id"] == "evt_1"

        scope = MemoryScope(workspace_id="ws1")
        memory_event = MemoryEvent(
            id="mem_1",
            kind=MemoryEventKind.INGEST,
            scope=scope,
            source="unit",
            source_id="doc_1",
            node_ids=["a"],
            edge_ids=["edge_ab"],
        )
        await backend.save_memory_event(memory_event)
        assert [e.id for e in await backend.list_memory_events(scope=scope)] == ["mem_1"]
        await backend.save_memory_events_batch(
            [
                MemoryEvent(
                    id="mem_2",
                    kind=MemoryEventKind.UPDATE,
                    scope=scope,
                    source="unit",
                    source_id="doc_2",
                ),
                MemoryEvent(
                    id="mem_3",
                    kind=MemoryEventKind.DELETE,
                    scope=scope,
                    source="unit",
                    source_id="doc_3",
                ),
            ]
        )
        event_ids = {e.id for e in await backend.list_memory_events(scope=scope, limit=10)}
        assert {"mem_1", "mem_2", "mem_3"}.issubset(event_ids)

        retrieval_event = RetrievalEvent(
            id="ret_1",
            query="alpha",
            scope=scope,
            returned_node_ids=["a"],
            signal=FeedbackSignal.SELECTED,
        )
        await backend.save_retrieval_event(retrieval_event)
        fetched = await backend.get_retrieval_event("ret_1")
        assert fetched is not None
        assert fetched.returned_node_ids == ["a"]

        await backend.save_memory_score(
            MemoryScore(scope_key=scope.key, node_id="a", access_count=1, score=0.2)
        )
        scores = await backend.list_memory_scores(scope_key=scope.key, node_ids=["a"])
        assert scores[0].score == 0.2
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_search_record_and_scoped_feedback_do_not_pollute_global_counts():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    node = await graph.add("Alpha policy", "Alpha beta content")
    scope = MemoryScope(user_id="u1")

    result = await graph.search("Alpha", record=True, scope=scope)
    assert result.event_id
    assert result.nodes

    await graph.record_feedback(
        result.event_id,
        node_ids=[node.id],
        signal=FeedbackSignal.SELECTED,
        scope=scope,
    )

    updated = await graph.get(node.id)
    assert updated is not None
    assert updated.success_count == 0
    assert updated.failure_count == 0
    local_score = await backend.get_memory_score(scope.key, node_id=node.id)
    assert local_score is not None
    assert local_score.access_count == 1

    await graph.record_feedback(
        result.event_id,
        node_ids=[node.id],
        signal=FeedbackSignal.TASK_SUCCESS,
        scope=scope,
    )

    globally_updated = await graph.get(node.id)
    assert globally_updated is not None
    assert globally_updated.success_count >= 1
    global_score = await backend.get_memory_score("global", node_id=node.id)
    assert global_score is not None
    assert global_score.success_count == 1


@pytest.mark.asyncio
async def test_task_success_feedback_reuses_retrieval_nodes_for_hebbian_edges():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    alpha = await graph.add("Alpha policy", "Alpha content")
    beta = await graph.add("Beta policy", "Beta content")
    scope = MemoryScope(user_id="u1")
    parent = RetrievalEvent(
        id="ret_success_pair",
        query="policy relationship",
        scope=scope,
        returned_node_ids=[alpha.id, beta.id],
    )
    await backend.save_retrieval_event(parent)

    feedback = await graph.record_feedback(
        parent.id,
        signal=FeedbackSignal.TASK_SUCCESS,
        scope=scope,
    )

    assert feedback.selected_node_ids == [alpha.id, beta.id]
    updated_alpha = await graph.get(alpha.id)
    updated_beta = await graph.get(beta.id)
    assert updated_alpha is not None
    assert updated_beta is not None
    assert updated_alpha.success_count == 1
    assert updated_beta.success_count == 1
    edges = await backend.get_edges(alpha.id, direction="both")
    related = [
        edge
        for edge in edges
        if {edge.source_id, edge.target_id} == {alpha.id, beta.id} and edge.kind == EdgeKind.RELATED
    ]
    assert related
    assert related[0].weight > 0
    local_edge_score = await backend.get_memory_score(scope.key, edge_id=related[0].id)
    assert local_edge_score is not None
    assert local_edge_score.access_count == 1
    assert local_edge_score.success_count == 1
    assert local_edge_score.score == pytest.approx(0.2)
    global_edge_score = await backend.get_memory_score("global", edge_id=related[0].id)
    assert global_edge_score is not None
    assert global_edge_score.access_count == 1
    assert global_edge_score.success_count == 1
    assert global_edge_score.score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_global_task_feedback_updates_node_and_edge_scores_once():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    alpha = await graph.add("Alpha policy", "Alpha content")
    beta = await graph.add("Beta policy", "Beta content")

    await graph.record_feedback(
        node_ids=[alpha.id, beta.id],
        signal=FeedbackSignal.TASK_SUCCESS,
    )

    global_alpha_score = await backend.get_memory_score("global", node_id=alpha.id)
    assert global_alpha_score is not None
    assert global_alpha_score.access_count == 1
    assert global_alpha_score.success_count == 1
    assert global_alpha_score.score == pytest.approx(0.2)
    edges = await backend.get_edges(alpha.id, direction="both")
    related = [
        edge
        for edge in edges
        if {edge.source_id, edge.target_id} == {alpha.id, beta.id} and edge.kind == EdgeKind.RELATED
    ]
    assert related
    global_edge_score = await backend.get_memory_score("global", edge_id=related[0].id)
    assert global_edge_score is not None
    assert global_edge_score.access_count == 1
    assert global_edge_score.success_count == 1
    assert global_edge_score.score == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_public_reinforce_records_feedback_ledger_and_scores():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    alpha = await graph.add("Alpha policy", "Alpha content")
    beta = await graph.add("Beta policy", "Beta content")

    await graph.reinforce([alpha.id, beta.id], success=True)

    retrieval_events = await backend.list_retrieval_events(limit=10)
    assert len(retrieval_events) == 1
    assert retrieval_events[0].signal == FeedbackSignal.TASK_SUCCESS
    assert retrieval_events[0].selected_node_ids == [alpha.id, beta.id]
    feedback_events = await backend.list_memory_events(
        kind=MemoryEventKind.FEEDBACK,
        limit=10,
    )
    assert len(feedback_events) == 1
    assert feedback_events[0].node_ids == [alpha.id, beta.id]
    assert feedback_events[0].properties["signal"] == str(FeedbackSignal.TASK_SUCCESS)
    global_alpha_score = await backend.get_memory_score("global", node_id=alpha.id)
    assert global_alpha_score is not None
    assert global_alpha_score.success_count == 1
    edges = await backend.get_edges(alpha.id, direction="both")
    related = [
        edge
        for edge in edges
        if {edge.source_id, edge.target_id} == {alpha.id, beta.id} and edge.kind == EdgeKind.RELATED
    ]
    assert related
    global_edge_score = await backend.get_memory_score("global", edge_id=related[0].id)
    assert global_edge_score is not None
    assert global_edge_score.success_count == 1


@pytest.mark.asyncio
async def test_implicit_feedback_does_not_create_hebbian_edges_or_counts():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    alpha = await graph.add("Alpha policy", "Alpha content")
    beta = await graph.add("Beta policy", "Beta content")
    scope = MemoryScope(user_id="u1")

    await graph.record_feedback(
        node_ids=[alpha.id, beta.id],
        signal=FeedbackSignal.SELECTED,
        scope=scope,
    )

    updated_alpha = await graph.get(alpha.id)
    updated_beta = await graph.get(beta.id)
    assert updated_alpha is not None
    assert updated_beta is not None
    assert updated_alpha.success_count == 0
    assert updated_alpha.failure_count == 0
    assert updated_beta.success_count == 0
    assert updated_beta.failure_count == 0
    edges = await backend.get_edges(alpha.id, direction="both")
    assert [edge for edge in edges if {edge.source_id, edge.target_id} == {alpha.id, beta.id}] == []


@pytest.mark.asyncio
async def test_ignored_feedback_is_weak_scope_local_signal_without_counts():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    alpha = await graph.add("Alpha policy", "Alpha content")
    beta = await graph.add("Beta policy", "Beta content")
    scope = MemoryScope(user_id="u1")

    feedback = await graph.record_feedback(
        node_ids=[alpha.id, beta.id],
        signal=FeedbackSignal.IGNORED,
        scope=scope,
    )

    assert feedback.success is None
    updated_alpha = await graph.get(alpha.id)
    updated_beta = await graph.get(beta.id)
    assert updated_alpha is not None
    assert updated_beta is not None
    assert updated_alpha.success_count == 0
    assert updated_alpha.failure_count == 0
    assert updated_beta.success_count == 0
    assert updated_beta.failure_count == 0
    local_score = await backend.get_memory_score(scope.key, node_id=alpha.id)
    assert local_score is not None
    assert local_score.access_count == 1
    assert local_score.success_count == 0
    assert local_score.failure_count == 0
    assert local_score.score == pytest.approx(-0.01)
    assert await backend.get_memory_score("global", node_id=alpha.id) is None
    edges = await backend.get_edges(alpha.id, direction="both")
    assert [edge for edge in edges if {edge.source_id, edge.target_id} == {alpha.id, beta.id}] == []


@pytest.mark.asyncio
async def test_explicit_negative_feedback_stays_scope_local_without_global_opt_in():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    node = await graph.add("Beta policy", "Beta content")
    scope = MemoryScope(user_id="u1")

    feedback = await graph.record_feedback(
        node_ids=[node.id],
        signal=FeedbackSignal.EXPLICIT_NEGATIVE,
        scope=scope,
    )

    assert feedback.success is False
    updated = await graph.get(node.id)
    assert updated is not None
    assert updated.failure_count == 0
    local_score = await backend.get_memory_score(scope.key, node_id=node.id)
    assert local_score is not None
    assert local_score.access_count == 1
    assert local_score.failure_count == 1
    assert local_score.score == pytest.approx(-0.25)
    assert await backend.get_memory_score("global", node_id=node.id) is None

    promoted_scope = MemoryScope(user_id="u1", promote_to_global=True)
    await graph.record_feedback(
        node_ids=[node.id],
        signal=FeedbackSignal.EXPLICIT_NEGATIVE,
        scope=promoted_scope,
    )

    globally_updated = await graph.get(node.id)
    assert globally_updated is not None
    assert globally_updated.failure_count == 1
    global_score = await backend.get_memory_score("global", node_id=node.id)
    assert global_score is not None
    assert global_score.failure_count == 1


@pytest.mark.asyncio
async def test_explicit_negative_feedback_scores_existing_relation_scope_locally():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    alpha = await graph.add("Alpha policy", "Alpha content")
    beta = await graph.add("Beta policy", "Beta content")
    await backend.save_edge(
        Edge(
            id="existing_relation",
            source_id=alpha.id,
            target_id=beta.id,
            kind=EdgeKind.RELATED,
            weight=1.0,
        )
    )
    scope = MemoryScope(user_id="u1")

    await graph.record_feedback(
        node_ids=[alpha.id, beta.id],
        signal=FeedbackSignal.EXPLICIT_NEGATIVE,
        scope=scope,
    )

    alpha_after = await graph.get(alpha.id)
    beta_after = await graph.get(beta.id)
    assert alpha_after is not None
    assert beta_after is not None
    assert alpha_after.failure_count == 0
    assert beta_after.failure_count == 0
    local_edge_score = await backend.get_memory_score(scope.key, edge_id="existing_relation")
    assert local_edge_score is not None
    assert local_edge_score.access_count == 1
    assert local_edge_score.failure_count == 1
    assert local_edge_score.score == pytest.approx(-0.25)
    assert await backend.get_memory_score("global", edge_id="existing_relation") is None
    edge = (await backend.get_edges(alpha.id, direction="outgoing"))[0]
    assert edge.weight == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_task_success_feedback_feeds_consolidation_promotion():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    node = await graph.add("Durable memory", "Successful task evidence")
    scope = MemoryScope(user_id="u1")
    parent = RetrievalEvent(
        id="ret_consolidate",
        query="durable memory",
        scope=scope,
        returned_node_ids=[node.id],
    )
    await backend.save_retrieval_event(parent)

    for _ in range(3):
        await graph.record_feedback(
            parent.id,
            signal=FeedbackSignal.TASK_SUCCESS,
            scope=scope,
        )

    before = await graph.get(node.id)
    assert before is not None
    assert before.level == ConsolidationLevel.L0_RAW
    assert before.access_count >= 3
    assert before.success_count >= 3

    result = await graph.consolidate()

    after = await graph.get(node.id)
    assert after is not None
    assert after.level == ConsolidationLevel.L1_SPRINT
    assert node.id in result.nodes_updated


@pytest.mark.asyncio
async def test_search_record_false_is_side_effect_free_for_ledgers():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    await graph.add("Alpha policy", "Alpha beta content")

    result = await graph.search("Alpha")

    assert result.nodes
    assert result.event_id == ""
    assert await backend.list_retrieval_events(limit=10) == []
    assert await backend.list_memory_events(kind=MemoryEventKind.RETRIEVAL, limit=10) == []


@pytest.mark.asyncio
async def test_search_record_true_writes_retrieval_and_memory_events():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    node = await graph.add("Alpha policy", "Alpha beta content")
    scope = MemoryScope(workspace_id="ws1")

    result = await graph.search("Alpha", record=True, scope=scope)

    assert result.event_id
    retrieval_events = await backend.list_retrieval_events(scope=scope, limit=10)
    assert [event.id for event in retrieval_events] == [result.event_id]
    assert retrieval_events[0].query == "Alpha"
    assert retrieval_events[0].returned_node_ids == [item.node.id for item in result.nodes]
    assert node.id in retrieval_events[0].returned_node_ids
    assert retrieval_events[0].properties["query"] == "Alpha"
    assert retrieval_events[0].properties["returned_count"] == str(len(result.nodes))
    assert retrieval_events[0].properties["total_candidates"] == str(result.total_candidates)
    assert "search_time_ms" in retrieval_events[0].properties
    memory_events = await backend.list_memory_events(
        kind=MemoryEventKind.RETRIEVAL,
        scope=scope,
        limit=10,
    )
    assert [event.source_id for event in memory_events] == [result.event_id]
    assert memory_events[0].node_ids == retrieval_events[0].returned_node_ids
    assert memory_events[0].properties == retrieval_events[0].properties


@pytest.mark.asyncio
async def test_graph_mutations_record_memory_events():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)

    node = await graph.add("Original", "Body", source="unit")
    ingest_events = await backend.list_memory_events(kind=MemoryEventKind.INGEST, limit=10)
    ingest = next(event for event in ingest_events if event.source_id == node.id)
    assert ingest.source == "unit"
    assert ingest.node_ids == [node.id]
    assert ingest.content_hash
    assert ingest.properties["operation"] == "SynapticGraph.add"

    updated = await graph.update(node.id, title="Updated", content="New body")
    assert updated is not None
    update_events = await backend.list_memory_events(kind=MemoryEventKind.UPDATE, limit=10)
    update = next(event for event in update_events if event.source_id == node.id)
    assert update.node_ids == [node.id]
    assert update.content_hash != ingest.content_hash
    assert update.properties["previous_content_hash"] == ingest.content_hash
    assert update.properties["changed_fields"] == "title,content"

    other = Node(id="other", title="Other")
    await backend.save_node(other)
    edge = await graph.link(node.id, other.id)

    assert await graph.remove(node.id) is True
    delete_events = await backend.list_memory_events(kind=MemoryEventKind.DELETE, limit=10)
    delete = next(event for event in delete_events if event.source_id == node.id)
    assert delete.node_ids == [node.id]
    assert edge.id in delete.edge_ids
    assert delete.content_hash == update.content_hash
    assert delete.properties["operation"] == "SynapticGraph.remove"


@pytest.mark.asyncio
async def test_graph_edge_mutations_record_memory_events():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    source = await graph.add("Source", "A")
    target = await graph.add("Target", "B")

    edge = await graph.link(source.id, target.id, kind=EdgeKind.RELATED, weight=0.4)
    edge_ingests = await backend.list_memory_events(kind=MemoryEventKind.INGEST, limit=20)
    edge_ingest = next(event for event in edge_ingests if event.source_id == edge.id)
    assert edge_ingest.node_ids == [source.id, target.id]
    assert edge_ingest.edge_ids == [edge.id]
    assert edge_ingest.content_hash
    assert edge_ingest.properties["operation"] == "SynapticGraph.link"
    assert edge_ingest.properties["edge_kind"] == str(EdgeKind.RELATED)

    updated = await graph.update_edge(
        source.id,
        target.id,
        kind=EdgeKind.RELATED,
        new_kind=EdgeKind.DEPENDS_ON,
        new_weight=0.9,
    )
    assert updated == 1
    edge_updates = await backend.list_memory_events(kind=MemoryEventKind.UPDATE, limit=20)
    edge_update = next(event for event in edge_updates if event.source_id == edge.id)
    assert edge_update.edge_ids == [edge.id]
    assert edge_update.content_hash != edge_ingest.content_hash
    previous_hashes = json.loads(edge_update.properties["previous_edge_hashes"])
    assert previous_hashes == {edge.id: edge_ingest.content_hash}
    assert edge_update.properties["operation"] == "SynapticGraph.update_edge"
    assert edge_update.properties["new_kind"] == str(EdgeKind.DEPENDS_ON)

    removed = await graph.unlink(source.id, target.id, kind=EdgeKind.DEPENDS_ON)
    assert removed == 1
    edge_deletes = await backend.list_memory_events(kind=MemoryEventKind.DELETE, limit=20)
    edge_delete = next(event for event in edge_deletes if event.source_id == edge.id)
    assert edge_delete.edge_ids == [edge.id]
    assert edge_delete.node_ids == [source.id, target.id]
    assert edge_delete.content_hash == edge_update.content_hash
    assert edge_delete.properties["operation"] == "SynapticGraph.unlink"


@pytest.mark.asyncio
async def test_scoped_failure_does_not_promote_to_global_without_opt_in():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    node = await graph.add("Beta policy", "Beta content")
    scope = MemoryScope(user_id="u1")

    result = await graph.search("Beta", record=True, scope=scope)
    assert result.event_id

    await graph.record_feedback(
        result.event_id,
        node_ids=[node.id],
        signal=FeedbackSignal.TASK_FAILURE,
        scope=scope,
    )

    updated = await graph.get(node.id)
    assert updated is not None
    assert updated.failure_count == 0
    local_score = await backend.get_memory_score(scope.key, node_id=node.id)
    assert local_score is not None
    assert local_score.failure_count == 1
    assert local_score.score < 0
    assert await backend.get_memory_score("global", node_id=node.id) is None

    promoted_scope = MemoryScope(user_id="u1", promote_to_global=True)
    await graph.record_feedback(
        result.event_id,
        node_ids=[node.id],
        signal=FeedbackSignal.TASK_FAILURE,
        scope=promoted_scope,
    )

    globally_updated = await graph.get(node.id)
    assert globally_updated is not None
    assert globally_updated.failure_count >= 1
    global_score = await backend.get_memory_score("global", node_id=node.id)
    assert global_score is not None
    assert global_score.failure_count == 1


@pytest.mark.asyncio
async def test_scope_boost_is_capped_without_reversing_base_relevance():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    high = Node(id="high", title="High relevance")
    boosted = Node(id="boosted", title="Boosted memory")
    await backend.save_node(high)
    await backend.save_node(boosted)
    await backend.save_memory_score(MemoryScore(scope_key=scope.key, node_id="boosted", score=1.0))
    result = SearchResult(
        query="policy",
        nodes=[
            ActivatedNode(node=high, activation=1.0, resonance=1.0),
            ActivatedNode(node=boosted, activation=0.96, resonance=0.96),
        ],
    )

    await graph._apply_scope_boost(result, scope)

    assert [item.node.id for item in result.nodes] == ["high", "boosted"]
    assert result.nodes[1].resonance <= result.nodes[0].resonance
    assert result.nodes[1].resonance == pytest.approx(1.0)
    assert result.nodes[1].resonance <= 0.96 * 1.10
    assert result.diagnostics["memory_scope_boosted_nodes"] == 1.0
    assert result.diagnostics["memory_scope_demoted_nodes"] == 0.0
    assert result.diagnostics["memory_scope_adjusted_nodes"] == 1.0
    assert result.diagnostics["memory_scope_node_score_hits"] == 1.0
    assert result.diagnostics["memory_scope_edge_score_hits"] == 0.0
    assert result.diagnostics["memory_scope_max_abs_boost"] == pytest.approx(0.10)
    assert result.diagnostics["memory_scope_order_clamps"] == 1.0


@pytest.mark.asyncio
async def test_scope_boost_uses_edge_score_for_endpoint_without_node_score():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    high = Node(id="edge_high", title="High relevance")
    boosted = Node(id="edge_boosted", title="Edge boosted memory")
    linked = Node(id="edge_linked", title="Linked memory")
    await backend.save_node(high)
    await backend.save_node(boosted)
    await backend.save_node(linked)
    await backend.save_edge(
        Edge(
            id="edge_boost_relation",
            source_id=boosted.id,
            target_id=linked.id,
            kind=EdgeKind.RELATED,
        )
    )
    await backend.save_memory_score(
        MemoryScore(scope_key=scope.key, edge_id="edge_boost_relation", score=1.0)
    )
    result = SearchResult(
        query="policy",
        nodes=[
            ActivatedNode(node=high, activation=1.0, resonance=1.0),
            ActivatedNode(node=boosted, activation=0.96, resonance=0.96),
        ],
    )

    await graph._apply_scope_boost(result, scope)

    assert await backend.get_memory_score(scope.key, node_id=boosted.id) is None
    assert [item.node.id for item in result.nodes] == [high.id, boosted.id]
    assert result.nodes[1].resonance == pytest.approx(1.0)
    assert result.nodes[1].resonance <= result.nodes[0].resonance
    assert result.nodes[1].resonance <= 0.96 * 1.10
    assert result.diagnostics["memory_scope_boosted_nodes"] == 1.0
    assert result.diagnostics["memory_scope_demoted_nodes"] == 0.0
    assert result.diagnostics["memory_scope_adjusted_nodes"] == 1.0
    assert result.diagnostics["memory_scope_node_score_hits"] == 0.0
    assert result.diagnostics["memory_scope_edge_score_hits"] == 1.0
    assert result.diagnostics["memory_scope_max_abs_boost"] == pytest.approx(0.10)
    assert result.diagnostics["memory_scope_order_clamps"] == 1.0


@pytest.mark.asyncio
async def test_scope_negative_score_can_demote_higher_relevance_candidate():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    demoted = Node(id="scope_demoted", title="Demoted memory")
    clean = Node(id="scope_clean", title="Clean memory")
    await backend.save_node(demoted)
    await backend.save_node(clean)
    await backend.save_memory_score(
        MemoryScore(scope_key=scope.key, node_id=demoted.id, score=-1.0)
    )
    result = SearchResult(
        query="policy",
        nodes=[
            ActivatedNode(node=demoted, activation=1.0, resonance=1.0),
            ActivatedNode(node=clean, activation=0.96, resonance=0.96),
        ],
    )

    await graph._apply_scope_boost(result, scope)

    assert [item.node.id for item in result.nodes] == [clean.id, demoted.id]
    assert result.nodes[0].resonance == pytest.approx(0.96)
    assert result.nodes[1].resonance == pytest.approx(0.90)
    assert result.diagnostics["memory_scope_boosted_nodes"] == 0.0
    assert result.diagnostics["memory_scope_demoted_nodes"] == 1.0
    assert result.diagnostics["memory_scope_adjusted_nodes"] == 1.0
    assert result.diagnostics["memory_scope_max_abs_boost"] == pytest.approx(0.10)
    assert result.diagnostics["memory_scope_order_clamps"] == 0.0


@pytest.mark.asyncio
async def test_search_applies_scope_boost_cap_on_public_path(monkeypatch):
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    high = Node(id="public_high", title="High relevance")
    boosted = Node(id="public_boosted", title="Boosted memory")
    await backend.save_node(high)
    await backend.save_node(boosted)
    await backend.save_memory_score(MemoryScore(scope_key=scope.key, node_id=boosted.id, score=1.0))

    class FakeEvidenceSearch:
        async def search(self, query: str, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                query=query,
                evidence=[
                    SimpleNamespace(node=high, score=1.0),
                    SimpleNamespace(node=boosted, score=0.96),
                ],
                scored=[high, boosted],
                elapsed_ms=1.0,
                timings_ms={},
            )

    monkeypatch.setattr(
        SynapticGraph,
        "_get_evidence_search",
        lambda self, active_reranker: FakeEvidenceSearch(),
    )

    result = await graph.search("policy", scope=scope, record=True)

    assert [item.node.id for item in result.nodes] == [high.id, boosted.id]
    assert result.nodes[1].resonance == pytest.approx(1.0)
    assert result.nodes[1].resonance <= result.nodes[0].resonance
    assert result.nodes[1].resonance <= 0.96 * 1.10
    assert result.event_id
    recorded = await backend.get_retrieval_event(result.event_id)
    assert recorded is not None
    assert recorded.returned_node_ids == [high.id, boosted.id]
    assert recorded.properties["memory_scope_boosted_nodes"] == "1.000000"
    assert recorded.properties["memory_scope_node_score_hits"] == "1.000000"
    assert recorded.properties["memory_scope_max_abs_boost"] == "0.100000"
    memory_events = await backend.list_memory_events(
        kind=MemoryEventKind.RETRIEVAL,
        scope=scope,
        limit=10,
    )
    memory_event = next(event for event in memory_events if event.source_id == result.event_id)
    assert memory_event.properties == recorded.properties


@pytest.mark.asyncio
async def test_scoped_search_uses_global_prior_without_reversing_relevance():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    high = Node(id="high", title="High relevance")
    globally_boosted = Node(id="global", title="Globally promoted memory")
    await backend.save_node(high)
    await backend.save_node(globally_boosted)
    await backend.save_memory_score(MemoryScore(scope_key="global", node_id="global", score=1.0))
    result = SearchResult(
        query="policy",
        nodes=[
            ActivatedNode(node=high, activation=1.0, resonance=1.0),
            ActivatedNode(node=globally_boosted, activation=0.96, resonance=0.96),
        ],
    )

    await graph._apply_scope_boost(result, scope)

    assert [item.node.id for item in result.nodes] == ["high", "global"]
    assert result.nodes[1].resonance == pytest.approx(1.0)
    assert result.nodes[1].resonance <= result.nodes[0].resonance


@pytest.mark.asyncio
async def test_search_applies_global_prior_half_weight_on_public_path(monkeypatch):
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    high = Node(id="public_global_high", title="High relevance")
    globally_boosted = Node(id="public_global_boosted", title="Global prior memory")
    await backend.save_node(high)
    await backend.save_node(globally_boosted)
    await backend.save_memory_score(
        MemoryScore(scope_key="global", node_id=globally_boosted.id, score=1.0)
    )

    class FakeEvidenceSearch:
        async def search(self, query: str, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                query=query,
                evidence=[
                    SimpleNamespace(node=high, score=1.0),
                    SimpleNamespace(node=globally_boosted, score=0.90),
                ],
                scored=[high, globally_boosted],
                elapsed_ms=1.0,
                timings_ms={},
            )

    monkeypatch.setattr(
        SynapticGraph,
        "_get_evidence_search",
        lambda self, active_reranker: FakeEvidenceSearch(),
    )

    result = await graph.search("policy", scope=scope)

    assert [item.node.id for item in result.nodes] == [high.id, globally_boosted.id]
    assert result.nodes[1].resonance == pytest.approx(0.90 * 1.05)
    assert result.nodes[1].resonance <= result.nodes[0].resonance


@pytest.mark.asyncio
async def test_scope_local_negative_can_counter_global_prior():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    clean = Node(id="clean", title="Clean memory")
    mixed = Node(id="mixed", title="Mixed feedback memory")
    await backend.save_node(clean)
    await backend.save_node(mixed)
    await backend.save_memory_score(MemoryScore(scope_key="global", node_id="mixed", score=1.0))
    await backend.save_memory_score(MemoryScore(scope_key=scope.key, node_id="mixed", score=-1.0))
    result = SearchResult(
        query="policy",
        nodes=[
            ActivatedNode(node=clean, activation=1.0, resonance=1.0),
            ActivatedNode(node=mixed, activation=0.98, resonance=0.98),
        ],
    )

    await graph._apply_scope_boost(result, scope)

    assert result.nodes[1].node.id == "mixed"
    assert result.nodes[1].resonance == pytest.approx(0.98 * 0.95)
    assert result.nodes[1].resonance < 0.98


@pytest.mark.asyncio
async def test_search_applies_high_confidence_suspect_signal_penalty(monkeypatch):
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    suspect = Node(id="suspect", title="Suspect memory")
    clean = Node(id="clean", title="Clean memory")
    signal = Node(
        id="sig_suspect",
        kind=NodeKind.OBSERVATION,
        title="Memory signal",
        tags=["_memory_signal", "_memory_suspect"],
        properties={
            "scope_key": scope.key,
            "node_ids": "suspect",
            "confidence": "1.0",
            "signal_kind": str(MemorySignalKind.POSSIBLE_CONFLICT),
        },
    )
    await backend.save_node(suspect)
    await backend.save_node(clean)
    await backend.save_node(signal)

    class FakeEvidenceSearch:
        async def search(self, query: str, **_: object) -> SimpleNamespace:
            return SimpleNamespace(
                query=query,
                evidence=[
                    SimpleNamespace(node=suspect, score=1.0),
                    SimpleNamespace(node=clean, score=0.98),
                ],
                scored=[suspect, clean],
                elapsed_ms=1.0,
                timings_ms={},
            )

    monkeypatch.setattr(
        SynapticGraph,
        "_get_evidence_search",
        lambda self, active_reranker: FakeEvidenceSearch(),
    )

    result = await graph.search("policy", scope=scope, record=True)

    assert [item.node.id for item in result.nodes] == ["clean", "suspect"]
    suspect_item = next(item for item in result.nodes if item.node.id == "suspect")
    assert suspect_item.resonance == pytest.approx(0.95)
    assert result.diagnostics["memory_signal_penalized_nodes"] == 1.0
    assert result.diagnostics["memory_signal_max_penalty"] == pytest.approx(0.05)
    assert result.diagnostics["memory_signal_penalized_node_ids"] == "suspect"
    assert result.diagnostics["memory_signal_source_ids"] == "sig_suspect"
    assert result.event_id
    recorded = await backend.get_retrieval_event(result.event_id)
    assert recorded is not None
    assert recorded.properties["memory_signal_penalized_node_ids"] == "suspect"
    assert recorded.properties["memory_signal_source_ids"] == "sig_suspect"


@pytest.mark.asyncio
async def test_memory_signal_penalty_resolves_edge_only_signal_targets():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    suspect = Node(id="edge_suspect", title="Suspect relation endpoint")
    linked = Node(id="edge_linked", title="Linked endpoint")
    clean = Node(id="edge_clean", title="Clean memory")
    await backend.save_node(suspect)
    await backend.save_node(linked)
    await backend.save_node(clean)
    await backend.save_edge(
        Edge(
            id="suspect_relation",
            source_id=suspect.id,
            target_id=linked.id,
            kind=EdgeKind.RELATED,
        )
    )
    await backend.save_node(
        Node(
            id="sig_edge_only",
            kind=NodeKind.OBSERVATION,
            title="Edge-only memory signal",
            tags=["_memory_signal", "_memory_suspect"],
            properties={
                "scope_key": scope.key,
                "edge_ids": "suspect_relation",
                "confidence": "1.0",
                "signal_kind": str(MemorySignalKind.REPEATED_FAILURE),
            },
        )
    )
    result = SearchResult(
        query="policy",
        nodes=[
            ActivatedNode(node=suspect, activation=1.0, resonance=1.0),
            ActivatedNode(node=clean, activation=0.98, resonance=0.98),
        ],
    )

    await graph._apply_memory_signal_penalties(result, scope=scope)

    assert [item.node.id for item in result.nodes] == [clean.id, suspect.id]
    suspect_item = next(item for item in result.nodes if item.node.id == suspect.id)
    assert suspect_item.resonance == pytest.approx(0.95)
    assert result.diagnostics["memory_signal_penalized_nodes"] == 1.0
    assert result.diagnostics["memory_signal_max_penalty"] == pytest.approx(0.05)
    assert result.diagnostics["memory_signal_penalized_node_ids"] == suspect.id
    assert result.diagnostics["memory_signal_source_ids"] == "sig_edge_only"
    assert result.diagnostics["memory_signal_edge_ids"] == "suspect_relation"


@pytest.mark.asyncio
async def test_memory_signal_penalty_ignores_low_confidence_and_other_scope():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    suspect = Node(id="suspect", title="Suspect memory")
    clean = Node(id="clean", title="Clean memory")
    await backend.save_node(suspect)
    await backend.save_node(clean)
    await backend.save_node(
        Node(
            id="sig_low_confidence",
            kind=NodeKind.OBSERVATION,
            title="Low confidence signal",
            tags=["_memory_signal", "_memory_suspect"],
            properties={
                "scope_key": scope.key,
                "node_ids": "suspect",
                "confidence": "0.69",
            },
        )
    )
    await backend.save_node(
        Node(
            id="sig_other_scope",
            kind=NodeKind.OBSERVATION,
            title="Other scope signal",
            tags=["_memory_signal", "_memory_suspect"],
            properties={
                "scope_key": "user:u2",
                "node_ids": "suspect",
                "confidence": "1.0",
            },
        )
    )
    result = SearchResult(
        query="policy",
        nodes=[
            ActivatedNode(node=suspect, activation=1.0, resonance=1.0),
            ActivatedNode(node=clean, activation=0.98, resonance=0.98),
        ],
    )

    await graph._apply_memory_signal_penalties(result, scope=scope)

    assert [item.node.id for item in result.nodes] == ["suspect", "clean"]
    assert result.nodes[0].resonance == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_memory_monitor_flags_suspect_memory_without_deleting_it():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    a = Node(id="a", title="A", kind=NodeKind.ENTITY)
    b = Node(id="b", title="B", kind=NodeKind.ENTITY)
    failed = Node(id="failed", title="Failed", failure_count=4, success_count=1)
    await backend.save_node(a)
    await backend.save_node(b)
    await backend.save_node(failed)
    await backend.save_edge(
        Edge(
            id="custom_openie_c1",
            source_id="a",
            target_id="b",
            kind=EdgeKind.CONTRADICTS,
            properties={"is_openie": "true", "confidence": "0.55"},
        )
    )

    signals = await graph.scan_memory_signals()
    kinds = {MemorySignalKind(str(signal.kind)) for signal in signals}
    assert MemorySignalKind.POSSIBLE_CONFLICT in kinds
    assert MemorySignalKind.LOW_CONFIDENCE_RELATION in kinds
    assert MemorySignalKind.REPEATED_FAILURE in kinds
    assert await backend.get_node("failed") is not None

    health = await graph.memory_health()
    assert health.suspect_count >= 3
    assert health.openie_artifact_count == 1
    assert {"a", "b", "failed"}.issubset(set(health.top_suspect_node_ids))
    assert "custom_openie_c1" in health.top_suspect_edge_ids
    assert health.top_suspect_node_counts["a"] == 2
    assert health.top_suspect_node_counts["b"] == 2
    assert health.top_suspect_node_counts["failed"] == 1
    assert health.top_suspect_edge_counts["custom_openie_c1"] == 2


@pytest.mark.asyncio
async def test_memory_health_summarizes_retrieval_ranking_diagnostics():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_boost",
            query="boost",
            scope=scope,
            properties={
                "memory_scope_boosted_nodes": "2.000000",
                "memory_scope_adjusted_nodes": "2.000000",
                "memory_scope_max_abs_boost": "0.100000",
                "memory_scope_max_positive_boost": "0.100000",
            },
        )
    )
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_demotion",
            query="demotion",
            scope=scope,
            properties={
                "memory_scope_demoted_nodes": "2.000000",
                "memory_scope_adjusted_nodes": "2.000000",
                "memory_scope_max_abs_boost": "0.080000",
                "memory_scope_max_demotion": "0.080000",
            },
        )
    )
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_penalty",
            query="penalty",
            scope=scope,
            properties={
                "memory_signal_penalized_nodes": "1.000000",
                "memory_signal_max_penalty": "0.050000",
                "memory_signal_penalized_node_ids": "node_penalized_a,node_penalized_b",
                "memory_signal_source_ids": "sig_penalty_a",
                "memory_signal_edge_ids": "edge_penalty_a",
            },
        )
    )
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_penalty_again",
            query="penalty again",
            scope=scope,
            properties={
                "memory_signal_penalized_nodes": "1.000000",
                "memory_signal_max_penalty": "0.030000",
                "memory_signal_penalized_node_ids": "node_penalized_a",
                "memory_signal_source_ids": "sig_penalty_a,sig_penalty_b",
                "memory_signal_edge_ids": "edge_penalty_a,edge_penalty_b",
            },
        )
    )
    await backend.save_retrieval_event(
        RetrievalEvent(
            id="ret_other_scope",
            query="other",
            scope=MemoryScope(user_id="u2"),
            properties={
                "memory_scope_boosted_nodes": "9.000000",
                "memory_signal_penalized_nodes": "9.000000",
            },
        )
    )
    for i in range(12):
        await backend.save_memory_score(
            MemoryScore(scope_key=scope.key, node_id=f"node_top_{i}", score=1.0)
        )
    await backend.save_memory_score(MemoryScore(scope_key=scope.key, edge_id="edge_top", score=0.5))
    await backend.save_memory_score(
        MemoryScore(scope_key=scope.key, node_id="node_demoted", score=-0.9)
    )
    await backend.save_memory_score(
        MemoryScore(scope_key=scope.key, node_id="node_less_demoted", score=-0.2)
    )
    await backend.save_memory_score(
        MemoryScore(scope_key=scope.key, node_id="node_neutral", score=0.0)
    )
    await backend.save_memory_score(
        MemoryScore(scope_key=scope.key, edge_id="edge_demoted", score=-0.7)
    )

    health = await graph.memory_health(scope=scope, persist_signals=False)

    assert health.retrieval_events == 4
    assert health.memory_boosted_retrieval_count == 1
    assert health.memory_demoted_retrieval_count == 1
    assert health.memory_adjusted_retrieval_count == 2
    assert health.memory_penalized_retrieval_count == 2
    assert health.memory_boosted_node_count == 2
    assert health.memory_demoted_node_count == 2
    assert health.memory_adjusted_node_count == 4
    assert health.memory_penalized_node_count == 2
    assert health.max_memory_scope_boost == pytest.approx(0.10)
    assert health.max_memory_scope_demotion == pytest.approx(0.08)
    assert health.max_memory_scope_adjustment == pytest.approx(0.10)
    assert health.max_memory_signal_penalty == pytest.approx(0.05)
    assert len(health.top_reinforced_node_ids) == 10
    assert health.top_reinforced_edge_ids == ["edge_top"]
    assert len(health.top_reinforced_node_scores) == 10
    assert health.top_reinforced_node_scores["node_top_0"] == pytest.approx(1.0)
    assert health.top_reinforced_edge_scores == {"edge_top": 0.5}
    assert health.top_demoted_node_ids == ["node_demoted", "node_less_demoted"]
    assert health.top_demoted_edge_ids == ["edge_demoted"]
    assert health.top_demoted_node_scores == {
        "node_demoted": -0.9,
        "node_less_demoted": -0.2,
    }
    assert health.top_demoted_edge_scores == {"edge_demoted": -0.7}
    assert health.top_penalty_signal_ids == ["sig_penalty_a", "sig_penalty_b"]
    assert health.top_penalized_node_ids == ["node_penalized_a", "node_penalized_b"]
    assert health.top_penalty_edge_ids == ["edge_penalty_a", "edge_penalty_b"]
    assert health.top_penalty_signal_counts == {"sig_penalty_a": 2, "sig_penalty_b": 1}
    assert health.top_penalized_node_counts == {
        "node_penalized_a": 2,
        "node_penalized_b": 1,
    }
    assert health.top_penalty_edge_counts == {"edge_penalty_a": 2, "edge_penalty_b": 1}


@pytest.mark.asyncio
async def test_memory_monitor_flags_recent_growth_and_reinforcement_signals_idempotently():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    now = time()
    since = now - 10
    old_entity = Node(
        id="old_entity",
        title="Old entity",
        kind=NodeKind.ENTITY,
        created_at=now - 100,
        updated_at=now,
    )
    new_entity = Node(
        id="new_entity",
        title="New entity",
        kind=NodeKind.ENTITY,
        created_at=now,
        updated_at=now,
    )
    await backend.save_node(old_entity)
    await backend.save_node(new_entity)
    await backend.save_edge(
        Edge(
            id="recent_reinforced_relation",
            source_id=old_entity.id,
            target_id=new_entity.id,
            kind=EdgeKind.RELATED,
            weight=1.2,
            properties={"support_count": "2"},
            created_at=now,
        )
    )

    first = await graph.scan_memory_signals(since=since)
    second = await graph.scan_memory_signals(since=since)
    by_kind = {MemorySignalKind(str(signal.kind)): signal for signal in first}

    assert MemorySignalKind.NEW_ENTITY in by_kind
    assert by_kind[MemorySignalKind.NEW_ENTITY].node_ids == [new_entity.id]
    assert MemorySignalKind.NEW_RELATION in by_kind
    assert by_kind[MemorySignalKind.NEW_RELATION].edge_ids == ["recent_reinforced_relation"]
    assert MemorySignalKind.RELATION_REINFORCED in by_kind
    assert by_kind[MemorySignalKind.RELATION_REINFORCED].edge_ids == ["recent_reinforced_relation"]
    assert {str(signal.kind) for signal in second}.issuperset(
        {
            str(MemorySignalKind.NEW_ENTITY),
            str(MemorySignalKind.NEW_RELATION),
            str(MemorySignalKind.RELATION_REINFORCED),
        }
    )
    signal_events = await backend.list_memory_events(
        kind=MemoryEventKind.SIGNAL,
        limit=10,
    )
    assert len(signal_events) == 3
    signal_nodes = await backend.list_nodes(kind=NodeKind.OBSERVATION, limit=10)
    lifecycle_nodes = [
        node
        for node in signal_nodes
        if node.properties.get("signal_kind")
        in {
            str(MemorySignalKind.NEW_ENTITY),
            str(MemorySignalKind.NEW_RELATION),
            str(MemorySignalKind.RELATION_REINFORCED),
        }
    ]
    assert len(lifecycle_nodes) == 3
    assert all("_memory_signal" in node.tags for node in lifecycle_nodes)
    assert all("_memory_suspect" not in node.tags for node in lifecycle_nodes)

    health = await graph.memory_health(since=since)
    assert health.signal_count == 3
    assert health.new_entity_count == 1
    assert health.new_relation_count == 1
    assert health.relation_reinforced_count == 1
    assert health.suspect_count == 0


@pytest.mark.asyncio
async def test_memory_monitor_records_signal_events_idempotently():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    failed = Node(id="failed_once", title="Failed once", failure_count=4, success_count=1)
    await backend.save_node(failed)

    first = await graph.scan_memory_signals(scope=scope)
    second = await graph.scan_memory_signals(scope=scope)

    repeated = [
        signal
        for signal in first
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.REPEATED_FAILURE
    ]
    assert len(repeated) == 1
    assert [
        signal
        for signal in second
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.REPEATED_FAILURE
    ]
    signal_events = await backend.list_memory_events(
        kind=MemoryEventKind.SIGNAL,
        scope=scope,
        limit=10,
    )
    assert len(signal_events) == 1
    event = signal_events[0]
    assert event.source == "memory_monitor"
    assert event.source_id == repeated[0].id
    assert failed.id in event.node_ids
    assert repeated[0].id in event.node_ids
    assert event.properties["signal_kind"] == str(MemorySignalKind.REPEATED_FAILURE)
    assert event.properties["scope_key"] == scope.key
    assert event.properties["node_ids"] == failed.id
    assert event.properties["confidence"] == str(repeated[0].confidence)
    assert event.properties["reason"] == repeated[0].reason
    signal_nodes = await backend.list_nodes(kind=NodeKind.OBSERVATION, limit=10)
    stored_signal = next(node for node in signal_nodes if node.id == repeated[0].id)
    assert "_memory_signal" in stored_signal.tags
    assert "_memory_suspect" in stored_signal.tags


@pytest.mark.asyncio
async def test_memory_monitor_flags_scope_score_repeated_failures_without_node_pollution():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    node = Node(id="scoped_failed", title="Scoped failed memory")
    await backend.save_node(node)
    await backend.save_memory_score(
        MemoryScore(
            scope_key=scope.key,
            node_id=node.id,
            access_count=4,
            success_count=0,
            failure_count=3,
            score=-0.75,
        )
    )

    signals = await graph.scan_memory_signals(scope=scope)
    repeated = [
        signal
        for signal in signals
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.REPEATED_FAILURE
        and node.id in signal.node_ids
    ]

    assert repeated
    assert repeated[0].properties["score_scope_key"] == scope.key
    stored = await backend.get_node(node.id)
    assert stored is not None
    assert stored.failure_count == 0
    signal_nodes = await backend.list_nodes(kind=NodeKind.OBSERVATION, limit=100)
    signal_node = next(
        signal_node
        for signal_node in signal_nodes
        if signal_node.properties.get("score_scope_key") == scope.key
    )
    assert "_memory_suspect" in signal_node.tags

    health = await graph.memory_health(scope=scope)
    assert health.repeated_failure_count >= 1


@pytest.mark.asyncio
async def test_memory_monitor_flags_strong_negative_scope_score_as_suspect():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    node = Node(id="strong_negative", title="Strong negative memory")
    await backend.save_node(node)
    await backend.save_memory_score(
        MemoryScore(
            scope_key=scope.key,
            node_id=node.id,
            access_count=1,
            success_count=0,
            failure_count=0,
            score=-0.75,
        )
    )

    signals = await graph.scan_memory_signals(scope=scope)
    repeated = [
        signal
        for signal in signals
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.REPEATED_FAILURE
        and node.id in signal.node_ids
    ]

    assert repeated
    assert repeated[0].properties["score_signal_type"] == "strong_negative_scope_score"
    assert repeated[0].properties["score_scope_key"] == scope.key
    assert repeated[0].properties["score"] == "-0.750000"
    signal_nodes = await backend.list_nodes(kind=NodeKind.OBSERVATION, limit=100)
    signal_node = next(
        signal_node
        for signal_node in signal_nodes
        if signal_node.properties.get("score_signal_type") == "strong_negative_scope_score"
    )
    assert "_memory_suspect" in signal_node.tags
    signal_events = await backend.list_memory_events(
        kind=MemoryEventKind.SIGNAL,
        scope=scope,
        limit=10,
    )
    event = next(event for event in signal_events if event.source_id == repeated[0].id)
    assert event.properties["score_signal_type"] == "strong_negative_scope_score"
    assert event.properties["score_scope_key"] == scope.key
    assert event.properties["score"] == "-0.750000"
    assert event.properties["node_ids"] == node.id
    assert event.properties["reason"] == repeated[0].reason

    health = await graph.memory_health(scope=scope)
    assert node.id in health.top_demoted_node_ids
    assert health.repeated_failure_count >= 1


@pytest.mark.asyncio
async def test_memory_monitor_carries_edge_score_signal_endpoints_into_penalty():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(user_id="u1")
    suspect = Node(id="edge_score_suspect", title="Edge score suspect")
    linked = Node(id="edge_score_linked", title="Edge score linked")
    clean = Node(id="edge_score_clean", title="Edge score clean")
    await backend.save_node(suspect)
    await backend.save_node(linked)
    await backend.save_node(clean)
    await backend.save_edge(
        Edge(
            id="edge_score_negative_relation",
            source_id=suspect.id,
            target_id=linked.id,
            kind=EdgeKind.RELATED,
        )
    )
    await backend.save_memory_score(
        MemoryScore(
            scope_key=scope.key,
            edge_id="edge_score_negative_relation",
            score=-0.8,
        )
    )

    signals = await graph.scan_memory_signals(scope=scope)
    edge_signal = next(
        signal
        for signal in signals
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.REPEATED_FAILURE
        and signal.edge_ids == ["edge_score_negative_relation"]
    )

    assert set(edge_signal.node_ids) == {suspect.id, linked.id}
    assert edge_signal.properties["score_signal_type"] == "strong_negative_scope_score"
    signal_node = await backend.get_node(edge_signal.id)
    assert signal_node is not None
    assert signal_node.properties["node_ids"] == f"{suspect.id},{linked.id}"

    result = SearchResult(
        query="edge score penalty",
        nodes=[
            ActivatedNode(node=suspect, activation=1.0, resonance=1.0),
            ActivatedNode(node=clean, activation=0.98, resonance=0.98),
        ],
    )
    await graph._apply_memory_signal_penalties(result, scope=scope)

    assert result.diagnostics["memory_signal_penalized_nodes"] == 1.0
    assert [item.node.id for item in result.nodes] == [clean.id, suspect.id]
    penalized = next(item for item in result.nodes if item.node.id == suspect.id)
    assert result.diagnostics["memory_signal_max_penalty"] == pytest.approx(0.0415)
    assert result.diagnostics["memory_signal_penalized_node_ids"] == suspect.id
    assert result.diagnostics["memory_signal_source_ids"] == edge_signal.id
    assert result.diagnostics["memory_signal_edge_ids"] == "edge_score_negative_relation"
    assert penalized.resonance == pytest.approx(0.9585)


@pytest.mark.asyncio
async def test_memory_monitor_flags_entity_property_conflicts_by_source():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    old = Node(
        id="entity_old",
        title="Alpha Policy",
        kind=NodeKind.ENTITY,
        source="doc_old",
        properties={"status": "active"},
    )
    new = Node(
        id="entity_new",
        title="Alpha Policy",
        kind=NodeKind.ENTITY,
        source="doc_new",
        properties={"status": "retired"},
    )
    await backend.save_node(old)
    await backend.save_node(new)

    signals = await graph.scan_memory_signals()
    conflict = next(
        signal
        for signal in signals
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.POSSIBLE_CONFLICT
        and signal.properties.get("conflict_type") == "entity_property_value"
    )

    assert set(conflict.node_ids) == {"entity_old", "entity_new"}
    assert conflict.properties["property_key"] == "status"
    assert conflict.properties["property_values"] == "active|retired"
    assert await backend.get_node("entity_old") is not None
    assert await backend.get_node("entity_new") is not None

    health = await graph.memory_health()
    assert health.conflict_signal_count >= 1


@pytest.mark.asyncio
async def test_memory_monitor_flags_superseded_target_as_stale_without_deleting():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    now = time()
    old = Node(
        id="old_policy",
        title="Old policy",
        updated_at=now - 400 * 24 * 3600,
    )
    new = Node(
        id="new_policy",
        title="New policy",
        updated_at=now,
    )
    await backend.save_node(old)
    await backend.save_node(new)
    await backend.save_edge(
        Edge(
            id="supersedes_policy",
            source_id=new.id,
            target_id=old.id,
            kind=EdgeKind.SUPERSEDES,
            created_at=now,
        )
    )

    signals = await graph.scan_memory_signals()
    stale = next(
        signal
        for signal in signals
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.STALE_MEMORY
        and signal.properties.get("stale_reason") == "superseded"
    )

    assert stale.node_ids == [old.id]
    assert stale.edge_ids == ["supersedes_policy"]
    assert stale.properties["superseding_node_id"] == new.id
    assert await backend.get_node(old.id) is not None

    health = await graph.memory_health()
    assert health.possible_supersession_count >= 1
    assert health.stale_signal_count >= 1
    assert health.suspect_count >= 1
    assert health.signal_kind_counts[str(MemorySignalKind.POSSIBLE_SUPERSESSION)] >= 1
    assert health.signal_kind_counts[str(MemorySignalKind.STALE_MEMORY)] >= 1


@pytest.mark.asyncio
async def test_memory_monitor_flags_semantic_extraction_drift_spike():
    backend = MemoryBackend()
    graph = SynapticGraph(backend)
    scope = MemoryScope(workspace_id="ws1")
    await backend.save_node(Node(id="a", title="A"))
    for i in range(3):
        await backend.save_memory_event(
            MemoryEvent(
                id=f"semantic_fail_{i}",
                kind=MemoryEventKind.SEMANTIC_EXTRACT,
                scope=scope,
                source="openie",
                source_id=f"chunk_{i}",
                node_ids=["a"],
                edge_ids=[f"edge_{i}"],
                confidence=0.3,
                properties={
                    "chunks_selected": "1",
                    "extraction_failures": "1",
                    "extractor": "OpenIELinker",
                    "model": "unstable-model",
                    "prompt_version": "v-drift",
                },
            )
        )

    signals = await graph.scan_memory_signals(scope=scope)
    drift = next(
        signal
        for signal in signals
        if MemorySignalKind(str(signal.kind)) == MemorySignalKind.DRIFT_SPIKE
    )

    assert drift.confidence >= 0.7
    signal_nodes = await backend.list_nodes(kind=NodeKind.OBSERVATION, limit=100)
    drift_node = next(
        node
        for node in signal_nodes
        if node.properties.get("signal_kind") == str(MemorySignalKind.DRIFT_SPIKE)
    )
    assert "_memory_suspect" in drift_node.tags
    assert drift_node.properties["failure_count"] == "3"
    assert drift_node.properties["attempt_count"] == "3"
    assert drift_node.properties["failure_rate"] == "1.000000"

    health = await graph.memory_health(scope=scope)
    assert health.drift_spike_count >= 1
    assert health.suspect_count >= 1
