"""Tests for SearchSession and SessionStore."""

from __future__ import annotations

from synaptic.search_session import SearchSession, SessionStore


# --- SearchSession basics ---


class TestSearchSessionBasics:
    def test_session_has_generated_id(self):
        s = SearchSession()
        assert s.session_id
        assert len(s.session_id) == 12

    def test_session_respects_supplied_id(self):
        s = SearchSession(session_id="my-custom-id")
        assert s.session_id == "my-custom-id"

    def test_budget_defaults(self):
        s = SearchSession()
        assert s.budget_tool_calls == 20
        assert s.tool_calls_used == 0
        assert s.budget_remaining() == 20

    def test_record_call_decrements_budget(self):
        s = SearchSession(budget_tool_calls=3)
        s.record_call()
        assert s.tool_calls_used == 1
        assert s.budget_remaining() == 2

    def test_is_exhausted_after_budget_used(self):
        s = SearchSession(budget_tool_calls=2)
        assert not s.is_exhausted()
        s.record_call()
        s.record_call()
        assert s.is_exhausted()
        # Overshooting still reports exhausted, not negative
        s.record_call()
        assert s.budget_remaining() == 0


# --- Seen tracking ---


class TestSeenTracking:
    def test_mark_seen_accumulates(self):
        s = SearchSession()
        s.mark_seen(["a", "b", "c"])
        assert s.has_seen("a")
        assert s.has_seen("c")
        assert not s.has_seen("d")

    def test_mark_seen_ignores_empty(self):
        s = SearchSession()
        s.mark_seen(["", None, "a"])  # type: ignore[list-item]
        assert s.seen_node_ids == {"a"}

    def test_filter_unseen_preserves_order(self):
        s = SearchSession()
        s.mark_seen(["b", "d"])
        assert s.filter_unseen(["a", "b", "c", "d", "e"]) == ["a", "c", "e"]

    def test_filter_unseen_drops_empties(self):
        s = SearchSession()
        assert s.filter_unseen(["", "x", None]) == ["x"]  # type: ignore[list-item]


# --- Query history ---


class TestQueryHistory:
    def test_record_query_appends(self):
        s = SearchSession()
        s.record_query("foo")
        s.record_query("bar")
        assert s.queries_tried == ["foo", "bar"]

    def test_record_query_deduplicates(self):
        s = SearchSession()
        s.record_query("foo")
        s.record_query("foo")
        assert s.queries_tried == ["foo"]

    def test_record_query_ignores_empty(self):
        s = SearchSession()
        s.record_query("")
        s.record_query("   ")
        assert s.queries_tried == []

    def test_record_query_strips_whitespace(self):
        s = SearchSession()
        s.record_query("  foo  ")
        assert s.queries_tried == ["foo"]


# --- Categories explored ---


class TestCategoriesExplored:
    def test_mark_categories_accumulates(self):
        s = SearchSession()
        s.mark_categories(["규정", "운영"])
        s.mark_categories(["운영", "조사"])
        assert s.categories_explored == {"규정", "운영", "조사"}

    def test_mark_categories_ignores_empty(self):
        s = SearchSession()
        s.mark_categories(["", None, "x"])  # type: ignore[list-item]
        assert s.categories_explored == {"x"}


# --- Facts scratchpad ---


class TestFacts:
    def test_set_and_retrieve_fact(self):
        s = SearchSession()
        s.set_fact("last_query_anchors", {"categories": ["규정"]})
        assert s.facts["last_query_anchors"] == {"categories": ["규정"]}

    def test_facts_overwrite(self):
        s = SearchSession()
        s.set_fact("key", 1)
        s.set_fact("key", 2)
        assert s.facts["key"] == 2


# --- Summary ---


class TestSummary:
    def test_summary_keys(self):
        s = SearchSession(budget_tool_calls=10)
        s.record_call()
        s.mark_seen(["a", "b"])
        s.record_query("foo")
        s.mark_categories(["규정"])

        summary = s.summary()
        assert summary["session_id"] == s.session_id
        assert summary["tool_calls_used"] == 1
        assert summary["budget_remaining"] == 9
        assert summary["seen_nodes"] == 2
        assert summary["queries_tried"] == 1
        assert summary["last_queries"] == ["foo"]
        assert summary["categories_explored"] == ["규정"]

    def test_last_queries_truncated_to_three(self):
        s = SearchSession()
        for q in ["q1", "q2", "q3", "q4", "q5"]:
            s.record_query(q)
        assert s.summary()["last_queries"] == ["q3", "q4", "q5"]


# --- SessionStore ---


class TestSessionStore:
    def test_create_returns_session(self):
        store = SessionStore()
        s = store.create()
        assert s.session_id in store
        assert len(store) == 1

    def test_create_with_existing_id_returns_existing(self):
        store = SessionStore()
        s1 = store.create(session_id="shared")
        s2 = store.create(session_id="shared")
        assert s1 is s2

    def test_get_returns_none_for_unknown_id(self):
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_get_or_create_creates_when_missing(self):
        store = SessionStore()
        s = store.get_or_create()
        assert s.session_id in store

    def test_get_or_create_returns_existing(self):
        store = SessionStore()
        s1 = store.create(session_id="abc")
        s2 = store.get_or_create(session_id="abc")
        assert s1 is s2

    def test_delete_removes_session(self):
        store = SessionStore()
        s = store.create()
        store.delete(s.session_id)
        assert s.session_id not in store
        assert len(store) == 0

    def test_delete_noop_on_unknown_id(self):
        store = SessionStore()
        store.delete("nonexistent")
        assert len(store) == 0

    def test_custom_budget_applied(self):
        store = SessionStore()
        s = store.create(budget_tool_calls=5)
        assert s.budget_tool_calls == 5
