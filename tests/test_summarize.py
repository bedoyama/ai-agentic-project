from __future__ import annotations

import pytest
from langchain.tools import BaseTool

from briefing.schemas import UrlSummary
from briefing.summarize import summarize_url
from briefing.tools import fetch_page, fetch_url_context, html_to_text


def test_url_summary_round_trip() -> None:
    original = UrlSummary(
        title="LangGraph overview",
        url="https://docs.langchain.com/oss/python/langgraph/overview",
        key_points=["Graphs hold state", "Edges can loop"],
        caveats=["Needs a checkpointer for human-in-the-loop"],
    )
    restored = UrlSummary.model_validate_json(original.model_dump_json())
    assert restored == original


def test_fetch_url_context_is_langchain_tool() -> None:
    assert isinstance(fetch_url_context, BaseTool)
    assert fetch_url_context.name == "fetch_url_context"


def test_html_to_text_strips_script() -> None:
    html = (
        "<html><head><script>alert(1)</script><title>Hi</title></head>"
        "<body><h1>Hello</h1><p>World</p></body></html>"
    )
    text = html_to_text(html)
    assert "Hello" in text
    assert "World" in text
    assert "alert" not in text


class _FakeStructured:
    def __init__(self, result: UrlSummary) -> None:
        self.result = result

    def invoke(self, _messages: object) -> UrlSummary:
        return self.result


class _FakeLLM:
    def __init__(self, result: UrlSummary) -> None:
        self.result = result
        self.seen_schema: object = None

    def with_structured_output(self, schema: object, method: str = "json_schema") -> _FakeStructured:
        self.seen_schema = schema
        assert method == "json_schema"
        return _FakeStructured(self.result)


def test_summarize_url_with_mocked_fetch() -> None:
    page = "LangGraph lets you build stateful agents with cycles."
    fake = UrlSummary(
        title="Wrong title from model",
        url="https://example.invalid/wrong",
        key_points=["Stateful agents"],
        caveats=[],
    )
    llm = _FakeLLM(fake)
    result = summarize_url(
        "https://docs.langchain.com/oss/python/langgraph/overview",
        fetch=lambda _url: page,
        llm=llm,  # type: ignore[arg-type]
    )
    assert result.key_points == ["Stateful agents"]
    assert result.url == "https://docs.langchain.com/oss/python/langgraph/overview"
    assert llm.seen_schema is UrlSummary


class _FailingUrlContextLLM:
    def bind_tools(self, _tools: object) -> _FailingUrlContextLLM:
        return self

    def invoke(self, _prompt: object) -> object:
        raise RuntimeError("url_context unavailable")


class _FakeHttpResponse:
    headers = {"content-type": "text/html; charset=utf-8"}
    text = "<html><body><h1>Fetched</h1><p>Via HTTP fallback.</p></body></html>"

    def raise_for_status(self) -> None:
        return None


class _FakeHttpClient:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def __enter__(self) -> _FakeHttpClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(self, url: str) -> _FakeHttpResponse:
        assert url.startswith("https://")
        return _FakeHttpResponse()


def test_fetch_page_falls_back_to_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("briefing.tools.httpx.Client", _FakeHttpClient)
    text = fetch_page("https://example.com/page", llm=_FailingUrlContextLLM())  # type: ignore[arg-type]
    assert "Fetched" in text
    assert "Via HTTP fallback" in text
