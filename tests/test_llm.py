"""Тесты разбора ответов LLM и формирования запросов."""

from __future__ import annotations

from app.pipeline.llm import LLMClient, extract_json_text, parse_json_object


def test_parse_plain_json():
    assert parse_json_object('{"a": 1}') == {"a": 1}


def test_parse_markdown_fence():
    raw = 'Вот результат:\n```json\n{"keep_segments": [{"start": 0, "end": 5}]}\n```\nГотово.'
    data = parse_json_object(raw)
    assert data["keep_segments"][0]["end"] == 5


def test_parse_json_with_trailing_text():
    data = parse_json_object('{"a": {"b": 2}} — пояснение')
    assert data["a"]["b"] == 2


def test_parse_json_with_trailing_comma():
    data = parse_json_object('{"a": 1,}')
    assert data["a"] == 1


def test_parse_broken_json_raises():
    try:
        parse_json_object("совсем не json")
    except Exception as exc:  # LLMError
        assert "JSON" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ожидалась ошибка разбора")


def test_extract_json_text_finds_nested_braces():
    text = 'prefix {"a": {"b": {"c": 1}}} suffix'
    assert extract_json_text(text) == '{"a": {"b": {"c": 1}}}'


def test_endpoint_builders():
    assert (
        LLMClient(base_url="http://127.0.0.1:1234/v1").endpoint
        == "http://127.0.0.1:1234/v1/chat/completions"
    )
    assert (
        LLMClient(base_url="http://127.0.0.1:11434").endpoint
        == "http://127.0.0.1:11434/v1/chat/completions"
    )
    assert (
        LLMClient(base_url="http://127.0.0.1:1234/v1/chat/completions").endpoint
        == "http://127.0.0.1:1234/v1/chat/completions"
    )


def test_headers_contain_bearer():
    client = LLMClient(base_url="http://x/v1", api_key="secret")
    assert client._headers()["Authorization"] == "Bearer secret"
    assert "Authorization" not in LLMClient(base_url="http://x/v1")._headers()
