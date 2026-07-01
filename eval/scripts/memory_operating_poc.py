#!/usr/bin/env python3
"""Deterministic smoke gate for the Synaptic memory operating layer.

This script avoids LLMs and remote embedding services. It exercises the
operating-layer contracts directly: retrieval ledgers, feedback ledgers,
scope-local/global node and edge reinforcement, feedback-fed consolidation,
edge provenance, pollution/growth signals, and the compact health report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from time import time
from typing import Any

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
    SearchResult,
)

HOME = Path.home() / "synaptic-eval"
DEFAULT_DB = HOME / "memory_operating_poc.db"
DEFAULT_RESULTS = HOME / "memory_operating_poc_results.json"


def _unlink_sqlite(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(f"{path}{suffix}")
        if candidate.exists():
            candidate.unlink()


def _round(value: float) -> float:
    return round(float(value), 6)


async def run_memory_operating_poc(args: argparse.Namespace) -> dict[str, Any]:
    if args.reset_db and args.db != Path(":memory:"):
        _unlink_sqlite(args.db)

    backend = SQLiteBackend(str(args.db))
    await backend.connect()
    try:
        graph = SynapticGraph(backend)
        scope = MemoryScope(
            workspace_id=args.workspace_id,
            user_id=args.user_id,
            session_id=args.session_id,
            domain=args.domain,
        )

        alpha = await graph.add(
            "Alpha retention policy",
            "Alpha policy keeps customer records for seven years and supersedes the old rule.",
        )
        beta = await graph.add(
            "Beta deletion policy",
            "Beta policy deletes temporary records after thirty days.",
        )
        failed = Node(
            id="poc_failed_memory",
            title="Repeatedly failed memory",
            content="This memory has repeated negative outcomes.",
            failure_count=4,
            success_count=1,
        )
        scoped_negative = Node(
            id="poc_scoped_negative_memory",
            title="Scoped negative memory",
            content="This memory receives only scoped negative feedback.",
        )
        scope_score_failed = Node(
            id="poc_scope_score_failed_memory",
            title="Scope score failed memory",
            content="This memory is repeatedly failed only in scoped score metadata.",
        )
        entity_conflict_old = Node(
            id="poc_entity_policy_old",
            kind=NodeKind.ENTITY,
            title="Policy State",
            source="poc_source_old",
            properties={"status": "active"},
        )
        entity_conflict_new = Node(
            id="poc_entity_policy_new",
            kind=NodeKind.ENTITY,
            title="Policy State",
            source="poc_source_new",
            properties={"status": "retired"},
        )
        superseded_policy = Node(
            id="poc_superseded_policy",
            title="Superseded policy",
            content="This policy has been replaced by newer evidence.",
            updated_at=time() - 400 * 24 * 3600,
        )
        superseding_policy = Node(
            id="poc_superseding_policy",
            title="Superseding policy",
            content="This policy supersedes the old policy.",
        )
        hebbian_left = Node(
            id="poc_hebbian_left",
            title="Hebbian left memory",
            content="This memory should be linked after successful pair feedback.",
        )
        hebbian_right = Node(
            id="poc_hebbian_right",
            title="Hebbian right memory",
            content="This memory should be linked after successful pair feedback.",
        )
        consolidation_candidate = Node(
            id="poc_consolidation_candidate",
            title="Consolidation candidate",
            content="Repeated task success should make this memory eligible for promotion.",
            level=ConsolidationLevel.L0_RAW,
        )
        await backend.save_node(failed)
        await backend.save_node(scoped_negative)
        await backend.save_node(scope_score_failed)
        await backend.save_node(entity_conflict_old)
        await backend.save_node(entity_conflict_new)
        await backend.save_node(superseded_policy)
        await backend.save_node(superseding_policy)
        await backend.save_node(hebbian_left)
        await backend.save_node(hebbian_right)
        await backend.save_node(consolidation_candidate)
        await backend.save_memory_score(
            MemoryScore(
                scope_key=scope.key,
                node_id=scope_score_failed.id,
                access_count=4,
                success_count=0,
                failure_count=3,
                score=-0.75,
            )
        )

        ingest_event = MemoryEvent(
            id="poc_ingest_event",
            kind=MemoryEventKind.INGEST,
            scope=scope,
            source="memory_operating_poc",
            source_id="poc_doc_1",
            node_ids=[
                alpha.id,
                beta.id,
                failed.id,
                scoped_negative.id,
                scope_score_failed.id,
                entity_conflict_old.id,
                entity_conflict_new.id,
                superseded_policy.id,
                superseding_policy.id,
                hebbian_left.id,
                hebbian_right.id,
                consolidation_candidate.id,
            ],
            confidence=1.0,
            properties={"mode": "deterministic"},
        )
        await backend.save_memory_event(ingest_event)

        semantic_event = MemoryEvent(
            id="poc_semantic_event",
            kind=MemoryEventKind.SEMANTIC_EXTRACT,
            scope=scope,
            source="memory_operating_poc",
            source_id="poc_doc_1",
            node_ids=[alpha.id, beta.id],
            confidence=0.72,
            properties={
                "extractor": "fixture",
                "model": "deterministic",
                "chunks_selected": "1",
                "extraction_failures": "0",
            },
        )
        await backend.save_memory_event(semantic_event)
        for i in range(3):
            await backend.save_memory_event(
                MemoryEvent(
                    id=f"poc_semantic_failure_{i}",
                    kind=MemoryEventKind.SEMANTIC_EXTRACT,
                    scope=scope,
                    source="openie",
                    source_id=f"poc_failed_chunk_{i}",
                    node_ids=[failed.id],
                    confidence=0.3,
                    properties={
                        "chunks_selected": "1",
                        "extraction_failures": "1",
                        "extractor": "fixture",
                        "model": "drift-fixture",
                        "prompt_version": "poc-drift-v1",
                    },
                )
            )

        openie_edge = Edge(
            id="openie_poc_alpha_beta",
            source_id=alpha.id,
            target_id=beta.id,
            kind=EdgeKind.RELATED,
            weight=1.0,
            properties={
                "source_event_id": semantic_event.id,
                "source_chunk_id": "poc_chunk_1",
                "extractor": "fixture",
                "model": "deterministic",
                "prompt_version": "poc-v1",
                "confidence": "0.55",
                "support_count": "1",
                "last_seen_at": str(time()),
                "is_openie": "true",
            },
        )
        conflict_edge = Edge(
            id="poc_conflict_alpha_beta",
            source_id=alpha.id,
            target_id=beta.id,
            kind=EdgeKind.CONTRADICTS,
            properties={"reason": "fixture conflict"},
        )
        await backend.save_edge(openie_edge)
        await backend.save_edge(conflict_edge)
        await backend.save_edge(
            Edge(
                id="poc_supersedes_policy_edge",
                source_id=superseding_policy.id,
                target_id=superseded_policy.id,
                kind=EdgeKind.SUPERSEDES,
            )
        )

        result = await graph.search("Alpha retention", limit=3, record=True, scope=scope)
        selected_node_ids = [item.node.id for item in result.nodes[:1]]
        selected_before_feedback = await graph.get(selected_node_ids[0])
        selected_feedback = await graph.record_feedback(
            result.event_id,
            node_ids=selected_node_ids,
            signal=FeedbackSignal.SELECTED,
            scope=scope,
        )
        selected_after_implicit = await graph.get(selected_node_ids[0])
        task_feedback = await graph.record_feedback(
            result.event_id,
            node_ids=selected_node_ids,
            signal=FeedbackSignal.TASK_SUCCESS,
            scope=scope,
        )
        selected_after_task = await graph.get(selected_node_ids[0])
        scoped_negative_before = await graph.get(scoped_negative.id)
        scoped_failure_feedback = await graph.record_feedback(
            result.event_id,
            node_ids=[scoped_negative.id],
            signal=FeedbackSignal.TASK_FAILURE,
            scope=scope,
        )
        scoped_negative_after = await graph.get(scoped_negative.id)
        await graph.record_feedback(
            node_ids=[hebbian_left.id, hebbian_right.id],
            signal=FeedbackSignal.TASK_SUCCESS,
            scope=scope,
        )
        hebbian_edges = await backend.get_edges(hebbian_left.id, direction="both")
        created_hebbian_edge = next(
            (
                edge
                for edge in hebbian_edges
                if {edge.source_id, edge.target_id} == {hebbian_left.id, hebbian_right.id}
                and edge.kind == EdgeKind.RELATED
            ),
            None,
        )
        hebbian_local_edge_score = (
            await backend.get_memory_score(scope.key, edge_id=created_hebbian_edge.id)
            if created_hebbian_edge is not None
            else None
        )
        hebbian_global_edge_score = (
            await backend.get_memory_score("global", edge_id=created_hebbian_edge.id)
            if created_hebbian_edge is not None
            else None
        )
        for _ in range(3):
            await graph.record_feedback(
                node_ids=[consolidation_candidate.id],
                signal=FeedbackSignal.TASK_SUCCESS,
                scope=scope,
            )
        before_consolidation = await backend.get_node(consolidation_candidate.id)
        consolidation_result = await graph.consolidate()
        after_consolidation = await backend.get_node(consolidation_candidate.id)

        roundtrip_edges = await backend.get_edges(alpha.id, direction="outgoing")
        roundtrip_openie = next(edge for edge in roundtrip_edges if edge.id == openie_edge.id)
        signals = await graph.scan_memory_signals(scope=scope)
        signal_kinds = sorted({str(signal.kind) for signal in signals})
        scope_score_failure_signal = next(
            (
                signal
                for signal in signals
                if str(signal.kind) == str(MemorySignalKind.REPEATED_FAILURE)
                and scope_score_failed.id in signal.node_ids
                and signal.properties.get("score_scope_key") == scope.key
            ),
            None,
        )
        property_conflict_signal = next(
            (
                signal
                for signal in signals
                if str(signal.kind) == str(MemorySignalKind.POSSIBLE_CONFLICT)
                and signal.properties.get("conflict_type") == "entity_property_value"
            ),
            None,
        )
        superseded_stale_signal = next(
            (
                signal
                for signal in signals
                if str(signal.kind) == str(MemorySignalKind.STALE_MEMORY)
                and superseded_policy.id in signal.node_ids
                and signal.properties.get("stale_reason") == "superseded"
            ),
            None,
        )
        health = await graph.memory_health(scope=scope)
        memory_events = await backend.list_memory_events(scope=scope, limit=1000)
        retrieval_events = await backend.list_retrieval_events(scope=scope, limit=1000)
        recorded_retrieval_event = next(
            (event for event in retrieval_events if event.id == result.event_id),
            None,
        )
        signal_events = [
            event for event in memory_events if str(event.kind) == str(MemoryEventKind.SIGNAL)
        ]
        signal_event_source_ids = {event.source_id for event in signal_events}
        scanned_signal_ids = {signal.id for signal in signals}
        local_score = await backend.get_memory_score(scope.key, node_id=selected_node_ids[0])
        global_score = await backend.get_memory_score("global", node_id=selected_node_ids[0])
        scoped_failure_score = await backend.get_memory_score(scope.key, node_id=scoped_negative.id)
        global_failure_score = await backend.get_memory_score("global", node_id=scoped_negative.id)
        global_prior = Node(
            id="poc_global_prior_memory",
            title="Globally promoted prior memory",
            content="This memory is promoted globally but has no local score.",
        )
        await backend.save_node(global_prior)
        await backend.save_memory_score(
            MemoryScore(scope_key="global", node_id=global_prior.id, score=1.0)
        )
        global_prior_result = SearchResult(
            query="global prior boost",
            nodes=[
                ActivatedNode(node=beta, activation=1.0, resonance=1.0),
                ActivatedNode(node=global_prior, activation=0.96, resonance=0.96),
            ],
        )
        await graph._apply_scope_boost(global_prior_result, scope)
        edge_boosted = Node(
            id="poc_edge_score_boosted_memory",
            title="Edge score boosted memory",
            content="This memory is boosted only through a reinforced relation score.",
        )
        edge_boost_linked = Node(
            id="poc_edge_score_linked_memory",
            title="Edge score linked memory",
            content="This memory provides the reinforced relation endpoint.",
        )
        await backend.save_node(edge_boosted)
        await backend.save_node(edge_boost_linked)
        await backend.save_edge(
            Edge(
                id="poc_edge_score_boost_relation",
                source_id=edge_boosted.id,
                target_id=edge_boost_linked.id,
                kind=EdgeKind.RELATED,
            )
        )
        await backend.save_memory_score(
            MemoryScore(
                scope_key=scope.key,
                edge_id="poc_edge_score_boost_relation",
                score=1.0,
            )
        )
        edge_boost_result = SearchResult(
            query="edge score boost",
            nodes=[
                ActivatedNode(node=beta, activation=1.0, resonance=1.0),
                ActivatedNode(node=edge_boosted, activation=0.96, resonance=0.96),
            ],
        )
        await graph._apply_scope_boost(edge_boost_result, scope)
        edge_boosted_item = next(
            item for item in edge_boost_result.nodes if item.node.id == edge_boosted.id
        )
        edge_boosted_node_score = await backend.get_memory_score(
            scope.key,
            node_id=edge_boosted.id,
        )
        failed_after_scan = await backend.get_node(failed.id)
        penalty_result = SearchResult(
            query="pollution penalty",
            nodes=[
                ActivatedNode(
                    node=failed_after_scan or failed,
                    activation=1.0,
                    resonance=1.0,
                ),
                ActivatedNode(
                    node=scoped_negative_after or scoped_negative,
                    activation=0.98,
                    resonance=0.98,
                ),
            ],
        )
        await graph._apply_memory_signal_penalties(penalty_result, scope=scope)
        penalized_failed = next(item for item in penalty_result.nodes if item.node.id == failed.id)
        edge_only_suspect = Node(
            id="poc_edge_only_signal_suspect",
            title="Edge-only signal endpoint",
            content="This endpoint should be demoted through a suspect relation signal.",
        )
        edge_only_linked = Node(
            id="poc_edge_only_signal_linked",
            title="Linked endpoint",
            content="This endpoint is connected by the suspect relation.",
        )
        edge_only_clean = Node(
            id="poc_edge_only_signal_clean",
            title="Clean edge-only comparator",
            content="This memory is not connected to the suspect relation.",
        )
        await backend.save_node(edge_only_suspect)
        await backend.save_node(edge_only_linked)
        await backend.save_node(edge_only_clean)
        await backend.save_edge(
            Edge(
                id="poc_edge_only_signal_relation",
                source_id=edge_only_suspect.id,
                target_id=edge_only_linked.id,
                kind=EdgeKind.RELATED,
            )
        )
        await backend.save_node(
            Node(
                id="poc_edge_only_signal_node",
                kind=NodeKind.OBSERVATION,
                title="Edge-only suspect signal",
                tags=["_memory_signal", "_memory_suspect"],
                properties={
                    "scope_key": scope.key,
                    "edge_ids": "poc_edge_only_signal_relation",
                    "confidence": "1.0",
                    "signal_kind": str(MemorySignalKind.REPEATED_FAILURE),
                },
            )
        )
        edge_only_penalty_result = SearchResult(
            query="edge-only pollution penalty",
            nodes=[
                ActivatedNode(node=edge_only_suspect, activation=1.0, resonance=1.0),
                ActivatedNode(node=edge_only_clean, activation=0.98, resonance=0.98),
            ],
        )
        await graph._apply_memory_signal_penalties(edge_only_penalty_result, scope=scope)
        edge_only_penalized = next(
            item for item in edge_only_penalty_result.nodes if item.node.id == edge_only_suspect.id
        )
        await graph._record_retrieval_event(edge_boost_result, scope=scope)
        await graph._record_retrieval_event(edge_only_penalty_result, scope=scope)
        ranking_health = await graph.memory_health(scope=scope)

        selected_before_success = (
            selected_before_feedback.success_count if selected_before_feedback else -1
        )
        selected_before_failure = (
            selected_before_feedback.failure_count if selected_before_feedback else -1
        )
        selected_implicit_success = (
            selected_after_implicit.success_count if selected_after_implicit else -1
        )
        selected_implicit_failure = (
            selected_after_implicit.failure_count if selected_after_implicit else -1
        )
        selected_task_success = selected_after_task.success_count if selected_after_task else -1
        scoped_negative_before_failure = (
            scoped_negative_before.failure_count if scoped_negative_before else -1
        )
        scoped_negative_after_failure = (
            scoped_negative_after.failure_count if scoped_negative_after else -1
        )

        gates = {
            "retrieval_event_recorded": bool(result.event_id)
            and any(event.id == result.event_id for event in retrieval_events),
            "retrieval_event_properties_recorded": (
                recorded_retrieval_event is not None
                and recorded_retrieval_event.properties.get("query") == result.query
                and recorded_retrieval_event.properties.get("returned_count")
                == str(len(result.nodes))
                and recorded_retrieval_event.properties.get("total_candidates")
                == str(result.total_candidates)
                and "search_time_ms" in recorded_retrieval_event.properties
            ),
            "feedback_events_recorded": {selected_feedback.id, task_feedback.id}.issubset(
                {event.id for event in retrieval_events}
            ),
            "implicit_feedback_did_not_update_global_counts": (
                selected_before_success == selected_implicit_success
                and selected_before_failure == selected_implicit_failure
            ),
            "task_success_promoted_global_counts": (
                selected_task_success > selected_implicit_success
            ),
            "scope_local_score_created": local_score is not None
            and local_score.access_count >= 2
            and local_score.success_count == 1,
            "global_score_created_only_after_task_success": global_score is not None
            and global_score.success_count == 1,
            "scoped_failure_stayed_local": (
                scoped_failure_score is not None
                and scoped_failure_score.failure_count == 1
                and global_failure_score is None
                and scoped_negative_after_failure == scoped_negative_before_failure
            ),
            "task_success_created_hebbian_edge": (
                created_hebbian_edge is not None and created_hebbian_edge.weight > 0
            ),
            "task_success_created_edge_scores": (
                hebbian_local_edge_score is not None
                and hebbian_local_edge_score.success_count == 1
                and hebbian_local_edge_score.score > 0
                and hebbian_global_edge_score is not None
                and hebbian_global_edge_score.success_count == 1
                and hebbian_global_edge_score.score > 0
            ),
            "feedback_counts_feed_consolidation": (
                before_consolidation is not None
                and before_consolidation.access_count >= 3
                and before_consolidation.success_count >= 3
                and after_consolidation is not None
                and after_consolidation.level == ConsolidationLevel.L1_SPRINT
                and consolidation_candidate.id in consolidation_result.nodes_updated
            ),
            "scope_score_repeated_failure_signal_created": (scope_score_failure_signal is not None),
            "entity_property_conflict_signal_created": (property_conflict_signal is not None),
            "superseded_target_stale_signal_created": (superseded_stale_signal is not None),
            "signal_events_recorded_idempotently": (
                bool(scanned_signal_ids)
                and scanned_signal_ids.issubset(signal_event_source_ids)
                and len(signal_events) == len(signal_event_source_ids)
            ),
            "global_prior_applied_without_reversing_scope_order": (
                [item.node.id for item in global_prior_result.nodes] == [beta.id, global_prior.id]
                and global_prior_result.nodes[1].resonance >= 0.999
                and global_prior_result.nodes[1].resonance <= global_prior_result.nodes[0].resonance
            ),
            "edge_score_boosted_endpoint_without_node_score": (
                edge_boosted_node_score is None
                and [item.node.id for item in edge_boost_result.nodes] == [beta.id, edge_boosted.id]
                and edge_boosted_item.resonance >= 0.999
                and edge_boosted_item.resonance <= edge_boost_result.nodes[0].resonance
            ),
            "memory_ranking_diagnostics_populated": (
                edge_boost_result.diagnostics.get("memory_scope_boosted_nodes") == 1.0
                and edge_boost_result.diagnostics.get("memory_scope_edge_score_hits") == 1.0
                and edge_boost_result.diagnostics.get("memory_scope_max_abs_boost", 0.0) > 0.0
                and penalty_result.diagnostics.get("memory_signal_penalized_nodes") == 1.0
                and penalty_result.diagnostics.get("memory_signal_max_penalty", 0.0) > 0.0
                and edge_only_penalty_result.diagnostics.get("memory_signal_penalized_nodes") == 1.0
            ),
            "health_summarizes_memory_ranking_diagnostics": (
                ranking_health.memory_boosted_retrieval_count >= 1
                and ranking_health.memory_penalized_retrieval_count >= 1
                and ranking_health.memory_boosted_node_count >= 1
                and ranking_health.memory_penalized_node_count >= 1
                and ranking_health.max_memory_scope_boost > 0.0
                and ranking_health.max_memory_signal_penalty > 0.0
            ),
            "health_reports_top_reinforced_edges": (
                "poc_edge_score_boost_relation" in ranking_health.top_reinforced_edge_ids
            ),
            "edge_provenance_roundtrip": (
                roundtrip_openie.properties.get("source_event_id") == semantic_event.id
                and roundtrip_openie.properties.get("model") == "deterministic"
                and roundtrip_openie.properties.get("is_openie") == "true"
            ),
            "pollution_signals_created": {
                str(MemorySignalKind.POSSIBLE_CONFLICT),
                str(MemorySignalKind.LOW_CONFIDENCE_RELATION),
                str(MemorySignalKind.REPEATED_FAILURE),
            }.issubset(set(signal_kinds)),
            "drift_spike_signal_created": str(MemorySignalKind.DRIFT_SPIKE) in set(signal_kinds),
            "suspect_memory_not_deleted": failed_after_scan is not None,
            "high_confidence_signal_demoted_suspect": (
                penalty_result.nodes[0].node.id == scoped_negative.id
                and penalized_failed.resonance < 0.98
                and penalized_failed.resonance >= 0.95
            ),
            "edge_only_signal_demoted_endpoint": (
                edge_only_penalty_result.nodes[0].node.id == edge_only_clean.id
                and edge_only_penalized.resonance < 0.98
                and edge_only_penalized.resonance >= 0.95
            ),
            "health_report_populated": (
                health.memory_events >= 4
                and health.retrieval_events >= 3
                and health.suspect_count >= 3
                and health.openie_artifact_count >= 1
            ),
        }
        passed = all(gates.values())
        payload = {
            "passed": passed,
            "db": str(args.db),
            "scope_key": scope.key,
            "gates": gates,
            "summary": {
                "returned_node_ids": [item.node.id for item in result.nodes],
                "alpha_node_id": alpha.id,
                "beta_node_id": beta.id,
                "retrieval_event_id": result.event_id,
                "retrieval_event_properties": (
                    dict(recorded_retrieval_event.properties) if recorded_retrieval_event else {}
                ),
                "selected_feedback_event_id": selected_feedback.id,
                "task_feedback_event_id": task_feedback.id,
                "scoped_failure_event_id": scoped_failure_feedback.id,
                "hebbian_edge_id": created_hebbian_edge.id if created_hebbian_edge else "",
                "consolidation_candidate_level": (
                    str(after_consolidation.level) if after_consolidation else ""
                ),
                "consolidation_nodes_updated": list(consolidation_result.nodes_updated),
                "memory_events": len(memory_events),
                "retrieval_events": len(retrieval_events),
                "signal_events": len(signal_events),
                "signal_kinds": signal_kinds,
                "local_score": asdict(local_score) if local_score else {},
                "global_score": asdict(global_score) if global_score else {},
                "scoped_failure_score": (
                    asdict(scoped_failure_score) if scoped_failure_score else {}
                ),
                "global_failure_score": (
                    asdict(global_failure_score) if global_failure_score else {}
                ),
                "hebbian_local_edge_score": (
                    asdict(hebbian_local_edge_score) if hebbian_local_edge_score else {}
                ),
                "hebbian_global_edge_score": (
                    asdict(hebbian_global_edge_score) if hebbian_global_edge_score else {}
                ),
                "scope_score_failure_signal": (
                    asdict(scope_score_failure_signal) if scope_score_failure_signal else {}
                ),
                "property_conflict_signal": (
                    asdict(property_conflict_signal) if property_conflict_signal else {}
                ),
                "superseded_stale_signal": (
                    asdict(superseded_stale_signal) if superseded_stale_signal else {}
                ),
                "global_prior_order": [item.node.id for item in global_prior_result.nodes],
                "global_prior_resonance": _round(global_prior_result.nodes[1].resonance),
                "edge_score_boost_order": [item.node.id for item in edge_boost_result.nodes],
                "edge_score_boosted_resonance": _round(edge_boosted_item.resonance),
                "edge_score_boost_diagnostics": dict(edge_boost_result.diagnostics),
                "health": asdict(health),
                "ranking_health": asdict(ranking_health),
                "penalty_order": [item.node.id for item in penalty_result.nodes],
                "penalized_failed_resonance": _round(penalized_failed.resonance),
                "penalty_diagnostics": dict(penalty_result.diagnostics),
                "edge_only_penalty_order": [
                    item.node.id for item in edge_only_penalty_result.nodes
                ],
                "edge_only_penalized_resonance": _round(edge_only_penalized.resonance),
                "edge_only_penalty_diagnostics": dict(edge_only_penalty_result.diagnostics),
                "search_time_ms": _round(result.search_time_ms),
                "timings_ms": {key: _round(value) for key, value in result.timings_ms.items()},
            },
        }
        return payload
    finally:
        await backend.close()


def write_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[results] wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--reset-db", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--workspace-id", default="memory-poc")
    parser.add_argument("--user-id", default="eval-user")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--domain", default="eval")
    parser.add_argument("--fail-on-gate", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    payload = await run_memory_operating_poc(args)
    write_results(args.results, payload)
    gate_state = "PASS" if payload["passed"] else "FAIL"
    print(
        "[memory-operating-poc] "
        f"{gate_state} scope={payload['scope_key']} "
        f"memory_events={payload['summary']['memory_events']} "
        f"retrieval_events={payload['summary']['retrieval_events']} "
        f"suspect_count={payload['summary']['health']['suspect_count']}"
    )
    return 0 if payload["passed"] or not args.fail_on_gate else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
