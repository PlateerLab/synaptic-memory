"""Safely write DeepSeek benchmark credentials to a gitignored local .env.

The script intentionally does not accept the API key as a command-line
argument, because command arguments are easy to leak through shell history and
process listings. It prompts with getpass and writes only the requested env var
to a local .env file with user-only permissions.
"""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = REPO_ROOT.parent / ".env"
DEFAULT_ENV_NAME = "DEEPSEEK_API_KEY"


def _split_env_line(line: str) -> tuple[str, str] | None:
    if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
        return None
    return key, value


def _quote_env_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _replace_env_value(lines: list[str], *, key: str, value: str) -> list[str]:
    new_line = f"{key}={_quote_env_value(value)}"
    updated: list[str] = []
    replaced = False
    for line in lines:
        parsed = _split_env_line(line)
        if parsed is not None and parsed[0] == key:
            if not replaced:
                updated.append(new_line)
                replaced = True
            continue
        updated.append(line)
    if not replaced:
        if updated and updated[-1].strip():
            updated.append("")
        updated.append(new_line)
    return updated


def _write_env_value(path: Path, *, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    updated = _replace_env_value(lines, key=key, value=value)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write("\n".join(updated) + "\n")
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _read_key(prompt: str) -> str:
    value = getpass.getpass(prompt).strip()
    if not value:
        raise SystemExit("Empty API key; nothing written.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-path",
        type=Path,
        default=DEFAULT_ENV_PATH,
        help="Local .env path to update. Default: parent workspace .env.",
    )
    parser.add_argument(
        "--env-name",
        default=DEFAULT_ENV_NAME,
        help="Environment variable name to write.",
    )
    args = parser.parse_args(argv)

    key = _read_key(f"{args.env_name}: ")
    _write_env_value(args.env_path, key=args.env_name, value=key)
    print(f"Wrote {args.env_name} to {args.env_path} with mode 0600.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
