# Contributing to Synaptic Memory

Thanks for your interest in contributing! This guide will help you get started.

## Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager

## Setup

```bash
git clone https://github.com/PlateerLab/synaptic-memory.git
cd synaptic-memory
uv sync --extra dev --extra sqlite --extra kuzu --extra qdrant --extra minio
```

## Running Tests

```bash
# Fast local suite (no external infra — uses Memory/SQLite/Kuzu embedded backends)
uv run pytest tests/ \
  --ignore=tests/test_backend_postgresql.py \
  --ignore=tests/test_backend_qdrant.py \
  --ignore=tests/test_backend_minio.py \
  --ignore=tests/test_backend_composite.py \
  --ignore=tests/benchmark \
  --ignore=tests/qa -v

# Full suite (requires Qdrant + MinIO + PostgreSQL running)
docker-compose up -d qdrant minio
uv run pytest tests/ -v
```

## Linting

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check --fix
uv run ruff format
```

Both checks must pass before merging. CI runs `ruff check` and `ruff format --check` automatically.

## Code Style

- **Ruff** handles all formatting and linting — no additional tools needed.
- **Type hints are required** for all function signatures.
- Keep imports sorted (Ruff handles this via `isort` rules).

## Pull Request Guidelines

1. **Create a feature branch** from `main` — do not push directly to `main`.
2. **All tests must pass** before requesting review.
3. **Use conventional commit messages**:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation only
   - `refactor:` code restructuring without behavior change
   - `test:` adding or updating tests
   - `chore:` maintenance tasks
4. **Keep PRs focused** — one feature or fix per PR.
5. **Add tests** for new functionality.

## Project Structure

```
src/synaptic/
├── graph.py           # Main facade (SynapticGraph)
├── search.py          # Hybrid search engine
├── agent_search.py    # Intent-based agent search
├── resonance.py       # Multi-axis scoring
├── hebbian.py         # Co-activation learning
├── consolidation.py   # Memory consolidation cascade
├── ontology.py        # Type hierarchy & constraints
├── activity.py        # Agent activity tracking
├── models.py          # Core data models
└── backends/          # Storage backends (Memory, SQLite, Kuzu, PostgreSQL, Qdrant, MinIO)
```

## Questions?

Open an issue on GitHub — we're happy to help.
