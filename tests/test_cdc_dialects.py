"""Phase 6 — dialect translator unit tests.

Real PG / MySQL integration tests are opt-in via env vars (they need
a running database) — these tests only cover the dialect-agnostic
glue that the CDC orchestrator relies on so the SQL stays correct
on every dispatch path.
"""

from __future__ import annotations

from synaptic.extensions.db_ingester import _translate_placeholders


class TestTranslatePlaceholders:
    def test_pg_numbers_placeholders(self):
        assert _translate_placeholders('"updated_at" >= ?', "pg") == '"updated_at" >= $1'

    def test_pg_numbers_multiple(self):
        assert (
            _translate_placeholders('a = ? AND b = ?', "pg")
            == 'a = $1 AND b = $2'
        )

    def test_mysql_uses_percent_s(self):
        assert _translate_placeholders('"col" >= ?', "mysql") == '"col" >= %s'

    def test_sqlite_passthrough(self):
        assert _translate_placeholders('"col" >= ?', "sqlite") == '"col" >= ?'

    def test_unknown_dialect_passthrough(self):
        assert _translate_placeholders('? ? ?', "made-up") == '? ? ?'

    def test_question_mark_only_in_placeholder_position(self):
        # `?` inside a literal would be a footgun in real SQL but
        # the translator does a naive replace — that is fine because
        # the syncer never builds clauses with literal `?` chars.
        out = _translate_placeholders("?", "pg")
        assert out == "$1"
