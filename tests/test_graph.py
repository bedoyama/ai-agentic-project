from __future__ import annotations

import json

from langchain.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool, tool
from langgraph.prebuilt import tools_condition

from briefing.graph import ASK_RECURSION_LIMIT, build_ask_graph
from briefing.tools import extract_search_sources, fetch_url, web_search


def test_agent_tools_are_langchain_tools() -> None:
    assert isinstance(web_search, BaseTool)
    assert web_search.name == "web_search"
    assert isinstance(fetch_url, BaseTool)
    assert fetch_url.name == "fetch_url"


def test_extract_search_sources_from_annotations() -> None:
    class _Response:
        text = "Pablo Galindo Salgado is the release manager."
        content_blocks = [
            {
                "type": "text",
                "text": "Pablo Galindo Salgado is the release manager.",
                "annotations": [
                    {
                        "type": "citation",
                        "url": "https://www.python.org/downloads/",
                        "title": "Python downloads",
                    }
                ],
            }
        ]
        response_metadata = {}

    sources = extract_search_sources(_Response())
    assert sources == [
        {"url": "https://www.python.org/downloads/", "title": "Python downloads"}
    ]


def test_extract_search_sources_from_text_urls() -> None:
    class _Response:
        text = "See https://peps.python.org/pep-0001/ for details."
        content_blocks = []
        response_metadata = {}

    sources = extract_search_sources(_Response())
    assert sources[0]["url"] == "https://peps.python.org/pep-0001/"


def test_tools_condition_routes_on_tool_calls() -> None:
    with_calls = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "python release manager"},
                        "id": "call_1",
                    }
                ],
            )
        ]
    }
    without_calls = {"messages": [AIMessage(content="Done.")]}
    assert tools_condition(with_calls) == "tools"
    assert tools_condition(without_calls) == "__end__"


@tool
def _fake_web_search(query: str) -> str:
    """Search the public web for current facts."""
    return json.dumps(
        {
            "query": query,
            "answer": "Pablo Galindo Salgado is a CPython release manager.",
            "sources": [
                {
                    "url": "https://www.python.org/downloads/",
                    "title": "Python downloads",
                }
            ],
        }
    )


class _FakeAskLLM:
    def bind_tools(self, _tools: object) -> _FakeAskLLM:
        return self

    def invoke(self, messages: list, config: object = None, **_kwargs: object) -> AIMessage:
        last = messages[-1]
        if getattr(last, "type", None) == "tool":
            return AIMessage(
                content=(
                    "Pablo Galindo Salgado is a current CPython release manager. "
                    "Source: https://www.python.org/downloads/"
                )
            )
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "_fake_web_search",
                    "args": {"query": "current Python release manager"},
                    "id": "call_1",
                }
            ],
        )


def test_ask_graph_searches_then_answers_with_source() -> None:
    graph = build_ask_graph(llm=_FakeAskLLM(), tools=[_fake_web_search])
    result = graph.invoke(
        {"messages": [HumanMessage(content="Who is the current Python release manager?")]},
        config={"recursion_limit": ASK_RECURSION_LIMIT},
    )
    texts = [
        getattr(message, "text", None) or str(getattr(message, "content", ""))
        for message in result["messages"]
    ]
    combined = "\n".join(texts)
    assert "Pablo Galindo Salgado" in combined
    assert "https://www.python.org/downloads/" in combined
    assert any(getattr(message, "tool_calls", None) for message in result["messages"])
