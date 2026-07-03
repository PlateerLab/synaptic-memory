"""User-facing build options for SynapticGraph one-line constructors."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from enum import StrEnum

DEFAULT_EMBED_MODEL = "qwen3-embedding:4b"
DEFAULT_RERANK_BACKEND = "vllm"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


class SynapticPreset(StrEnum):
    """Named option bundles for the high-level graph constructors."""

    LOCAL = "local"
    RAG = "rag"
    AGENT = "agent"
    SCALE = "scale"


@dataclass(frozen=True, slots=True)
class GraphBuildOptions:
    """Options shared by ``SynapticGraph.from_*`` constructors.

    The defaults are intentionally local and LLM-free. External embedding
    and reranking endpoints are only enabled when a caller supplies them
    directly or via ``SYNAPTIC_*`` environment variables in an endpoint-aware
    preset.
    """

    embed_url: str | None = None
    embed_model: str = DEFAULT_EMBED_MODEL
    rerank_url: str | None = None
    rerank_backend: str = DEFAULT_RERANK_BACKEND
    rerank_model: str = DEFAULT_RERANK_MODEL
    openie_enabled: bool = False
    connect: bool = False

    @classmethod
    def local(cls) -> GraphBuildOptions:
        """No external services, no mutation beyond ingest."""

        return cls()

    @classmethod
    def rag(cls) -> GraphBuildOptions:
        """Endpoint-aware RAG preset driven by ``SYNAPTIC_*`` env vars."""

        return cls(
            embed_url=_env_str("SYNAPTIC_EMBED_URL"),
            embed_model=_env_str("SYNAPTIC_EMBED_MODEL") or DEFAULT_EMBED_MODEL,
            rerank_url=_env_str("SYNAPTIC_RERANK_URL"),
            rerank_backend=_env_str("SYNAPTIC_RERANK_BACKEND") or DEFAULT_RERANK_BACKEND,
            rerank_model=_env_str("SYNAPTIC_RERANK_MODEL") or DEFAULT_RERANK_MODEL,
        )

    @classmethod
    def agent(cls) -> GraphBuildOptions:
        """RAG preset plus deterministic component bridging for exploration."""

        return replace(cls.rag(), connect=True)

    @classmethod
    def scale(cls) -> GraphBuildOptions:
        """Endpoint-aware preset for large ingests; indexing is configured separately."""

        return cls.rag()

    @classmethod
    def from_preset(cls, preset: SynapticPreset | str | None = None) -> GraphBuildOptions:
        """Build options from a preset name.

        ``None`` maps to ``local`` so existing constructor defaults remain
        deterministic and dependency-free.
        """

        selected = _coerce_preset(preset)
        if selected == SynapticPreset.LOCAL:
            return cls.local()
        if selected == SynapticPreset.RAG:
            return cls.rag()
        if selected == SynapticPreset.AGENT:
            return cls.agent()
        if selected == SynapticPreset.SCALE:
            return cls.scale()
        raise AssertionError(f"unhandled preset: {selected!r}")

    def with_overrides(
        self,
        *,
        embed_url: str | None = None,
        embed_model: str | None = None,
        rerank_url: str | None = None,
        rerank_backend: str | None = None,
        rerank_model: str | None = None,
        openie_enabled: bool | None = None,
        connect: bool | None = None,
    ) -> GraphBuildOptions:
        """Return a copy with explicit constructor kwargs applied."""

        updates: dict[str, object] = {}
        if embed_url is not None:
            updates["embed_url"] = embed_url
        if embed_model is not None:
            updates["embed_model"] = embed_model
        if rerank_url is not None:
            updates["rerank_url"] = rerank_url
        if rerank_backend is not None:
            updates["rerank_backend"] = rerank_backend
        if rerank_model is not None:
            updates["rerank_model"] = rerank_model
        if openie_enabled is not None:
            updates["openie_enabled"] = openie_enabled
        if connect is not None:
            updates["connect"] = connect
        return replace(self, **updates)


def resolve_graph_build_options(
    *,
    preset: SynapticPreset | str | None = None,
    options: GraphBuildOptions | None = None,
    embed_url: str | None = None,
    embed_model: str | None = None,
    rerank_url: str | None = None,
    rerank_backend: str | None = None,
    rerank_model: str | None = None,
    openie_enabled: bool | None = None,
    connect: bool | None = None,
) -> GraphBuildOptions:
    """Resolve constructor options with explicit kwargs taking precedence."""

    if preset is not None and options is not None:
        msg = "Pass either preset=... or options=..., not both"
        raise ValueError(msg)
    base = options if options is not None else GraphBuildOptions.from_preset(preset)
    return base.with_overrides(
        embed_url=embed_url,
        embed_model=embed_model,
        rerank_url=rerank_url,
        rerank_backend=rerank_backend,
        rerank_model=rerank_model,
        openie_enabled=openie_enabled,
        connect=connect,
    )


def _coerce_preset(preset: SynapticPreset | str | None) -> SynapticPreset:
    if preset is None:
        return SynapticPreset.LOCAL
    if isinstance(preset, SynapticPreset):
        return preset
    try:
        return SynapticPreset(str(preset).lower())
    except ValueError as exc:
        allowed = ", ".join(p.value for p in SynapticPreset)
        msg = f"Unknown Synaptic preset {preset!r}; expected one of: {allowed}"
        raise ValueError(msg) from exc


def _env_str(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None
