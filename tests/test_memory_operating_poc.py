"""Tests for the deterministic memory operating PoC."""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "eval" / "scripts" / "memory_operating_poc.py"
_SPEC = importlib.util.spec_from_file_location("memory_operating_poc", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
memory_operating_poc = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = memory_operating_poc
_SPEC.loader.exec_module(memory_operating_poc)

run_memory_operating_poc = memory_operating_poc.run_memory_operating_poc
write_results = memory_operating_poc.write_results


@pytest.mark.asyncio
async def test_memory_operating_poc_passes_and_writes_results(tmp_path: Path) -> None:
    args = Namespace(
        db=tmp_path / "memory_poc.db",
        results=tmp_path / "memory_poc_results.json",
        reset_db=True,
        workspace_id="ws",
        user_id="user",
        session_id="session",
        domain="domain",
        fail_on_gate=True,
    )

    payload = await run_memory_operating_poc(args)
    write_results(args.results, payload)

    assert payload["passed"] is True
    assert all(payload["gates"].values())
    assert payload["scope_key"] == "session:session"
    assert payload["summary"]["memory_events"] >= 4
    assert payload["summary"]["retrieval_events"] >= 3
    assert payload["summary"]["health"]["suspect_count"] >= 3
    assert payload["summary"]["hebbian_local_edge_score"]["success_count"] == 1
    assert payload["summary"]["hebbian_global_edge_score"]["success_count"] == 1
    assert args.results.exists()
