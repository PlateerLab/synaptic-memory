"""Friendly ``synaptic-mcp`` entrypoint with optional-extra guidance."""

from __future__ import annotations

import sys

from synaptic.mcp import __version__

HELP = """Usage: synaptic-mcp [OPTIONS]

Start the Synaptic Memory MCP server.

Options:
  --db PATH                  SQLite database path for the graph (default: knowledge.db)
  --dsn DSN                  PostgreSQL backend for the graph itself
  --source-dsn DSN           Default source database for CDC sync tools
  --embed-url URL            Embedding API base URL (OpenAI-compatible)
  --embed-model NAME         Embedding model name
  --rerank-url URL           Cross-encoder reranker server base URL
  --rerank-backend NAME      Reranker wire format: vllm, ollama, or tei
  --rerank-model NAME        Reranker model name
  --llm-url URL              LLM API base URL for knowledge_ask
  --llm-model NAME           LLM model name
  --llm-api-key KEY          API key for --llm-url
  --vector-min-cosine FLOAT  Absolute noise floor for vector cascade
  --vector-relative-drop FLOAT
                             Relative vector cutoff below the top hit
  --version                  Show version
  -h, --help                 Show this help

Install the MCP extra before running the server:
  pip install "synaptic-memory[mcp]"
"""


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv

    if "--version" in args:
        print(f"synaptic-mcp {__version__}")
        return 0
    if "--help" in args or "-h" in args:
        print(HELP)
        return 0

    try:
        from synaptic.mcp.server import main as server_main
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            print(
                "synaptic-mcp requires the MCP extra.\n"
                'Install with: pip install "synaptic-memory[mcp]"',
                file=sys.stderr,
            )
            return 2
        raise

    old_argv = sys.argv
    if argv is not None:
        sys.argv = [old_argv[0], *argv]
    try:
        try:
            server_main()
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
