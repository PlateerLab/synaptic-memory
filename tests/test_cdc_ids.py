"""Unit tests for the deterministic row-ID helper.

Phase 1 of the CDC implementation. Validates:

- Same ``(source_url, table, pk)`` always produces the same node ID.
- Cosmetic differences in the source URL (case, trailing slash,
  password rotation) collapse to the same canonical form.
- 16-hex output collides extremely rarely on synthetic large-scale
  inputs (smoke check, not a formal collision proof).
- Output is hex / safe to use as a SQLite TEXT primary key.
"""

from __future__ import annotations

from synaptic.extensions.cdc.ids import (
    deterministic_row_id,
    normalize_source_url,
)


class TestNormalizeSourceURL:
    def test_lowercases_scheme_and_host(self):
        assert normalize_source_url("Postgres://user@HOST:5432/db") == \
            normalize_source_url("postgresql://user@host:5432/db".replace("postgresql", "postgres"))

    def test_strips_trailing_slash(self):
        a = normalize_source_url("postgres://user@host/db/")
        b = normalize_source_url("postgres://user@host/db")
        assert a == b

    def test_drops_password(self):
        a = normalize_source_url("postgres://user:password@host/db")
        b = normalize_source_url("postgres://user:rotated_pw_2026@host/db")
        assert a == b

    def test_preserves_port(self):
        a = normalize_source_url("postgres://user@host:5432/db")
        b = normalize_source_url("postgres://user@host:5433/db")
        assert a != b

    def test_preserves_path(self):
        a = normalize_source_url("postgres://host/db1")
        b = normalize_source_url("postgres://host/db2")
        assert a != b

    def test_handles_bare_path(self):
        # SQLite-style bare path
        assert normalize_source_url("/var/data/graph.db") == "/var/data/graph.db"

    def test_handles_empty(self):
        assert normalize_source_url("") == ""


class TestDeterministicRowID:
    def test_same_input_same_output(self):
        a = deterministic_row_id("postgres://h/d", "products", "P001")
        b = deterministic_row_id("postgres://h/d", "products", "P001")
        assert a == b

    def test_returns_16_hex_chars(self):
        node_id = deterministic_row_id("postgres://h/d", "products", "P001")
        assert len(node_id) == 16
        assert all(c in "0123456789abcdef" for c in node_id)

    def test_different_pk_different_id(self):
        a = deterministic_row_id("postgres://h/d", "products", "P001")
        b = deterministic_row_id("postgres://h/d", "products", "P002")
        assert a != b

    def test_different_table_different_id(self):
        a = deterministic_row_id("postgres://h/d", "products", "P001")
        b = deterministic_row_id("postgres://h/d", "reviews", "P001")
        assert a != b

    def test_different_source_url_different_id(self):
        a = deterministic_row_id("postgres://h/db1", "products", "P001")
        b = deterministic_row_id("postgres://h/db2", "products", "P001")
        assert a != b

    def test_password_change_does_not_change_id(self):
        a = deterministic_row_id("postgres://u:pw1@h/d", "products", "P001")
        b = deterministic_row_id("postgres://u:pw2@h/d", "products", "P001")
        assert a == b

    def test_integer_pk_works(self):
        # int PKs are coerced to str() — no TypeError
        a = deterministic_row_id("postgres://h/d", "products", 123)
        b = deterministic_row_id("postgres://h/d", "products", "123")
        assert a == b

    def test_bigger_digest_size(self):
        node_id = deterministic_row_id("postgres://h/d", "products", "P001", digest_size=12)
        assert len(node_id) == 24
        assert all(c in "0123456789abcdef" for c in node_id)

    def test_no_collision_synthetic_50k(self):
        """Smoke test: 50k synthetic PKs should produce 50k unique IDs.

        Not a formal collision proof — just guards against regressions
        like hashing only the table name.
        """
        ids = {
            deterministic_row_id("postgres://h/d", "t", f"row_{i}")
            for i in range(50_000)
        }
        assert len(ids) == 50_000
