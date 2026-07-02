"""Tests for the local DeepSeek .env configuration helper."""

from __future__ import annotations

import importlib.util
import stat
import sys
from pathlib import Path

HELPER_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "ablation" / "configure_deepseek_env.py"
)
SPEC = importlib.util.spec_from_file_location("configure_deepseek_env", HELPER_PATH)
assert SPEC is not None and SPEC.loader is not None
helper = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = helper
SPEC.loader.exec_module(helper)


def test_replace_env_value_preserves_other_lines_and_dedupes() -> None:
    lines = [
        "# local secrets",
        "OTHER_KEY=keep",
        "DEEPSEEK_API_KEY=old",
        "DEEPSEEK_API_KEY=older",
    ]

    updated = helper._replace_env_value(lines, key="DEEPSEEK_API_KEY", value="new-secret")

    assert updated == [
        "# local secrets",
        "OTHER_KEY=keep",
        'DEEPSEEK_API_KEY="new-secret"',
    ]


def test_write_env_value_uses_user_only_permissions(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"

    helper._write_env_value(env_path, key="DEEPSEEK_API_KEY", value='sk-test"quoted')

    assert env_path.read_text(encoding="utf-8") == ('DEEPSEEK_API_KEY="sk-test\\"quoted"\n')
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600


def test_main_does_not_print_secret(tmp_path: Path, monkeypatch, capsys) -> None:
    env_path = tmp_path / ".env"
    value = "value-for-test"
    monkeypatch.setattr(helper, "_read_key", lambda prompt: value)

    assert helper.main(["--env-path", str(env_path)]) == 0

    captured = capsys.readouterr()
    assert value not in captured.out
    assert value not in captured.err
    assert value in env_path.read_text(encoding="utf-8")
