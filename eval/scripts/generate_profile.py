"""Auto-generate a DomainProfile TOML from sample data.

Domain-agnostic wrapper around ``synaptic.extensions.profile_generator``.
Reads samples from a JSONL or CSV file, calls the generator (with or
without an LLM), and writes the resulting profile as TOML.

Usage examples::

    # Rule-based only (no LLM, fast, offline)
    uv run python eval/scripts/generate_profile.py \\
        --name krra_auto \\
        --samples eval/data/parsed/krra/documents.jsonl \\
        --field content \\
        --categories-field category \\
        --output eval/data/profiles/krra_auto.toml

    # With Ollama LLM
    uv run python eval/scripts/generate_profile.py \\
        --name krra_auto \\
        --samples eval/data/parsed/krra/documents.jsonl \\
        --field content \\
        --categories-field category \\
        --llm ollama --llm-model qwen3:4b \\
        --output eval/data/profiles/krra_auto.toml

    # From CSV rows (assort data)
    uv run python eval/scripts/generate_profile.py \\
        --name assort_auto \\
        --samples eval/data/raw/assort/reviews.csv \\
        --field review_content \\
        --output eval/data/profiles/assort_auto.toml

The script never fails on LLM errors — it downgrades to rule-based
output and writes the profile anyway, so the overall workflow stays
resilient to model/network issues.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from synaptic.extensions.ontology_classifier import OntologyClassifier
from synaptic.extensions.profile_generator import ProfileGenerator


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--name", required=True, help="Profile name (written to TOML)")
    p.add_argument(
        "--samples",
        type=Path,
        required=True,
        help="Path to JSONL or CSV file containing samples",
    )
    p.add_argument(
        "--field",
        default=None,
        help="Field name to extract as sample text (default: auto-detect)",
    )
    p.add_argument(
        "--categories-field",
        default=None,
        help="Optional field name holding category labels",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max samples sent to LLM (default: 20)",
    )
    p.add_argument(
        "--sample-limit",
        type=int,
        default=200,
        help="Max rows loaded from input file (default: 200)",
    )
    p.add_argument(
        "--embedder",
        choices=["none", "ollama", "openai"],
        default="none",
        help="Embedding provider for OntologyClassifier (default: none)",
    )
    p.add_argument(
        "--embedder-model",
        default=None,
        help="Embedding model name (default: qwen3-embedding:4b for ollama)",
    )
    p.add_argument(
        "--embedder-base-url",
        default=None,
        help="Override embedder base URL",
    )
    p.add_argument(
        "--classifier-threshold",
        type=float,
        default=0.35,
        help="Min cosine similarity for classifier confidence (default: 0.35)",
    )
    p.add_argument(
        "--llm",
        choices=["none", "ollama", "openai", "anthropic"],
        default="none",
        help="LLM provider to use (default: none = rule-based + classifier only)",
    )
    p.add_argument(
        "--llm-model",
        default=None,
        help="LLM model name (default depends on provider)",
    )
    p.add_argument(
        "--llm-base-url",
        default=None,
        help="Override base URL (Ollama/OpenAI-compatible)",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output TOML path",
    )
    return p.parse_args()


def _auto_detect_field(row: dict) -> str | None:
    """Pick a reasonable text field from a sample row.

    Prefers ``content``, ``text``, ``body``, ``title`` in that order.
    Returns ``None`` if no text-like field is present.
    """
    for key in ("content", "text", "body", "description", "title", "review_content"):
        if key in row and isinstance(row[key], str) and row[key].strip():
            return key
    # Fall back: first string field
    for k, v in row.items():
        if isinstance(v, str) and len(v.strip()) >= 10:
            return k
    return None


def _load_samples(
    path: Path,
    field: str | None,
    categories_field: str | None,
    limit: int,
) -> tuple[list[str], list[str]]:
    """Load sample strings (and optional category labels) from a file.

    Supports JSONL and CSV based on extension. Any other suffix is
    treated as plain text (one sample per line).
    """
    samples: list[str] = []
    categories: list[str] = []

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if not line.strip():
                    continue
                row = json.loads(line)
                if field is None:
                    field = _auto_detect_field(row)
                    if field is None:
                        continue
                    print(f"  (auto-detected field: {field!r})")
                text = row.get(field, "")
                if isinstance(text, str) and text.strip():
                    samples.append(text)
                    if categories_field:
                        cat = row.get(categories_field, "")
                        if isinstance(cat, str):
                            categories.append(cat)
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                if field is None:
                    field = _auto_detect_field(row)
                    if field is None:
                        continue
                    print(f"  (auto-detected field: {field!r})")
                text = row.get(field, "")
                if isinstance(text, str) and text.strip():
                    samples.append(text)
                    if categories_field:
                        cat = row.get(categories_field, "")
                        if isinstance(cat, str):
                            categories.append(cat)
    else:
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if line.strip():
                    samples.append(line.strip())

    return samples, categories


def _build_embedder(args: argparse.Namespace):
    """Construct an ``EmbeddingProvider`` for OntologyClassifier, or ``None``.

    ``ollama`` uses the OpenAI-compatible ``/v1/embeddings`` endpoint
    Ollama exposes since 0.1.30 — no extra flag needed, just point at
    ``localhost:11434``. The embedder is injected into the classifier
    at profile-generation time; it never touches the ingestion path.
    """
    if args.embedder == "none":
        return None

    # Import locally so callers without aiohttp don't pay the cost
    from synaptic.extensions.embedder import OpenAIEmbeddingProvider

    if args.embedder == "ollama":
        return OpenAIEmbeddingProvider(
            api_base=args.embedder_base_url or "http://localhost:11434/v1",
            model=args.embedder_model or "qwen3-embedding:4b",
            api_key="ollama",  # ignored by Ollama but required by OpenAI client
        )
    if args.embedder == "openai":
        return OpenAIEmbeddingProvider(
            api_base=args.embedder_base_url or "https://api.openai.com/v1",
            model=args.embedder_model or "text-embedding-3-small",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
    return None


def _build_llm(args: argparse.Namespace):
    """Construct an ``LLMProvider`` from CLI args, or return ``None``."""
    if args.llm == "none":
        return None
    if args.llm == "ollama":
        from synaptic.extensions.llm_provider import OllamaLLMProvider

        return OllamaLLMProvider(
            base_url=args.llm_base_url or "http://localhost:11434",
            model=args.llm_model or "qwen3:4b",
        )
    if args.llm == "openai":
        from synaptic.extensions.llm_provider import OpenAILLMProvider

        return OpenAILLMProvider(
            api_base=args.llm_base_url or "https://api.openai.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            model=args.llm_model or "gpt-4o-mini",
        )
    if args.llm == "anthropic":
        from synaptic.extensions.llm_provider import AnthropicLLMProvider

        return AnthropicLLMProvider(
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            model=args.llm_model or "claude-sonnet-4-20250514",
        )
    return None


async def main() -> int:
    args = _parse_args()

    if not args.samples.exists():
        print(f"ERROR: samples file not found: {args.samples}")
        return 1

    print(
        f"Loading samples from {args.samples.relative_to(REPO_ROOT) if args.samples.is_absolute() else args.samples}"
    )
    samples, categories = _load_samples(
        args.samples, args.field, args.categories_field, args.sample_limit
    )
    print(f"  Loaded {len(samples)} samples, {len(categories)} categories")

    if not samples:
        print("ERROR: no samples loaded — check --field or file format")
        return 1

    embedder = _build_embedder(args)
    classifier: OntologyClassifier | None = None
    if embedder is not None:
        classifier = OntologyClassifier(
            embedder=embedder,
            threshold=args.classifier_threshold,
        )
        print(
            f"  Embedder: {args.embedder}"
            + (f" ({args.embedder_model})" if args.embedder_model else "")
        )
    else:
        print("  Embedder: none (classifier disabled)")

    llm = _build_llm(args)
    print(f"  LLM: {args.llm}" + (f" ({args.llm_model})" if args.llm_model else ""))

    generator = ProfileGenerator(
        classifier=classifier,
        llm=llm,
        max_samples=args.max_samples,
    )

    print("\nGenerating profile...")
    profile = await generator.generate(
        name=args.name,
        samples=samples,
        categories=categories if categories else None,
    )

    print("\n--- Generated profile ---")
    print(f"  name:              {profile.name}")
    print(f"  locale:            {profile.locale}")
    print(f"  stopwords_extra:   {len(profile.stopwords_extra)} terms")
    print(f"  ontology_hints:    {len(profile.ontology_hints)} entries")
    print(f"  metadata_patterns: {len(profile.metadata_strip_patterns)}")
    print(f"  reference_patterns:{len(profile.reference_patterns)}")
    print(f"  entity_patterns:   {len(profile.entity_hint_patterns)}")
    if profile.stopwords_extra:
        preview = sorted(profile.stopwords_extra)[:10]
        print(f"  stopwords preview: {preview}")
    if profile.ontology_hints:
        hints_preview = list(profile.ontology_hints.items())[:5]
        print(f"  hints preview:     {hints_preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile.save(args.output)
    print(f"\n✓ Profile written → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
