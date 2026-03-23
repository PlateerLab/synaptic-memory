"""LLM providers — generate text completions for semantic processing.

Supports JSON-mode generation:
- Ollama (/api/generate with format: "json")
- OpenAI-compatible (/v1/chat/completions with response_format)
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Generate text completions from an LLM."""

    async def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str: ...


class OllamaLLMProvider:
    """Ollama native generation endpoint (/api/generate).

    Forces JSON output via ``format: "json"``.

    Usage:
        provider = OllamaLLMProvider(
            base_url="http://localhost:11434",
            model="qwen3:0.6b",
        )
        result = await provider.generate(
            system="You are a helpful assistant. Respond in JSON.",
            user="Summarize this text.",
        )
    """

    __slots__ = ("_base_url", "_model", "_timeout")

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        *,
        model: str = "qwen3:0.6b",
        timeout: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        """Generate a JSON completion via Ollama /api/generate."""
        import aiohttp

        url = f"{self._base_url}/api/generate"
        payload = {
            "model": self._model,
            "system": system,
            "prompt": user,
            "format": "json",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    msg = f"Ollama generate error {resp.status}: {body[:200]}"
                    raise RuntimeError(msg)
                data = await resp.json()

        return data["response"]  # type: ignore[no-any-return]


class OpenAILLMProvider:
    """OpenAI-compatible chat completion provider.

    Forces JSON output via ``response_format: {"type": "json_object"}``.

    Works with any server implementing the /v1/chat/completions endpoint:
    - OpenAI:    api_base="https://api.openai.com/v1", model="gpt-4o-mini"
    - vLLM:      api_base="http://localhost:8000/v1", model="Qwen/Qwen2.5-7B-Instruct"
    - llama.cpp: api_base="http://localhost:8080/v1", model="default"
    - Ollama:    api_base="http://localhost:11434/v1", model="qwen3:0.6b"

    Usage:
        provider = OpenAILLMProvider(
            api_base="https://api.openai.com/v1",
            api_key="sk-...",
            model="gpt-4o-mini",
        )
        result = await provider.generate(
            system="You are a helpful assistant. Respond in JSON.",
            user="Summarize this text.",
        )
    """

    __slots__ = ("_api_base", "_api_key", "_model", "_timeout")

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        *,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        timeout: int = 120,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        """Generate a JSON completion via OpenAI-compatible chat API."""
        import aiohttp

        url = f"{self._api_base}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    msg = f"LLM API error {resp.status}: {body[:200]}"
                    raise RuntimeError(msg)
                data = await resp.json()

        return data["choices"][0]["message"]["content"]  # type: ignore[no-any-return]


class AnthropicLLMProvider:
    """Anthropic Messages API provider.

    Usage:
        provider = AnthropicLLMProvider(api_key="sk-ant-...")
        result = await provider.generate(
            system="You are a helpful assistant. Respond in JSON.",
            user="Classify this document.",
        )
    """

    __slots__ = ("_api_key", "_model", "_timeout")

    def __init__(
        self,
        *,
        api_key: str = "",
        model: str = "claude-sonnet-4-20250514",
        timeout: int = 120,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout

    async def generate(self, *, system: str, user: str, max_tokens: int = 1024) -> str:
        """Generate a completion via Anthropic Messages API."""
        import aiohttp

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    msg = f"Anthropic API error {resp.status}: {body[:200]}"
                    raise RuntimeError(msg)
                data = await resp.json()

        return data["content"][0]["text"]  # type: ignore[no-any-return]
