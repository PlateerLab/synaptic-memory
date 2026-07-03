from __future__ import annotations

import json
from pathlib import Path

import pytest

from synaptic.cli import mcp as mcp_cli
from synaptic.cli import quickstart


def test_quickstart_json_memory_backend(capsys: pytest.CaptureFixture[str]) -> None:
    rc = quickstart.main(["--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["backend"] == "memory"
    assert payload["node_count"] >= 5
    assert payload["queries"]
    assert payload["queries"][0]["hits"]


def test_quickstart_json_sqlite_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("aiosqlite")
    db_path = tmp_path / "quickstart.db"

    rc = quickstart.main(["--json", "--db", str(db_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["backend"] == "sqlite"
    assert payload["db"] == str(db_path)
    assert payload["node_count"] >= 5
    assert db_path.exists()


def test_quickstart_custom_query_and_preset(capsys: pytest.CaptureFixture[str]) -> None:
    rc = quickstart.main(["--json", "--preset", "local", "--query", "USB-C desk lamp"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert [item["query"] for item in payload["queries"]] == ["USB-C desk lamp"]
    assert payload["queries"][0]["hits"]


def test_mcp_help_does_not_require_mcp_extra(capsys: pytest.CaptureFixture[str]) -> None:
    rc = mcp_cli.main(["--help"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Usage: synaptic-mcp" in captured.out
    assert "synaptic-memory[mcp]" in captured.out


def test_mcp_version_uses_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    rc = mcp_cli.main(["--version"])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip().startswith("synaptic-mcp 0.")
