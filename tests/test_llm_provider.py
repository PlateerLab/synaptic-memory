"""Tests for LLM provider request-shaping helpers."""

from __future__ import annotations

from synaptic.extensions.llm_provider import _openai_response_format, _should_retry_json_object


def test_openai_response_format_defaults_to_json_object():
    assert _openai_response_format(None) == {"type": "json_object"}


def test_openai_response_format_wraps_plain_schema():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    fmt = _openai_response_format(schema)

    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "synaptic_response"
    assert fmt["json_schema"]["schema"] == schema
    assert fmt["json_schema"]["strict"] is True


def test_openai_response_format_accepts_pre_wrapped_schema():
    wrapped = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}}}
    assert _openai_response_format(wrapped) is wrapped


def test_openai_json_schema_fallback_only_for_schema_client_errors():
    assert _should_retry_json_object({"type": "json_schema"}, 400) is True
    assert _should_retry_json_object({"type": "json_schema"}, 422) is True
    assert _should_retry_json_object({"type": "json_schema"}, 500) is False
    assert _should_retry_json_object({"type": "json_object"}, 400) is False
