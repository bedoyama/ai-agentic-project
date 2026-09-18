from __future__ import annotations

import json

from briefing.graph import RESEARCH_RECURSION_LIMIT, build_research_graph
from briefing.nodes import after_grade, parse_search_payload
from briefing.schemas import EvidenceGrade, Subquestions
from briefing.state import (
    DEFAULT_MAX_LOOPS,
    initial_research_state,
    merge_sources,
)


def test_merge_sources_dedupes_by_url() -> None:
    left = [{"url": "https://a.example", "title": "A"}]
    right = [
        {"url": "https://a.example", "title": "A updated", "snippet": "hi"},
        {"url": "https://b.example", "title": "B"},
    ]
    merged = merge_sources(left, right)
    by_url = {item["url"]: item for item in merged}
    assert by_url["https://a.example"]["title"] == "A updated"
    assert by_url["https://a.example"]["snippet"] == "hi"
    assert "https://b.example" in by_url


def test_after_grade_routes() -> None:
    weak = initial_research_state("q")
    weak["grade_relevant"] = False
    weak["loop_count"] = 1
    assert after_grade(weak) == "rewrite"

    strong = dict(weak)
    strong["grade_relevant"] = True
    assert after_grade(strong) == "write_briefing"

    exhausted = dict(weak)
    exhausted["loop_count"] = DEFAULT_MAX_LOOPS
    assert after_grade(exhausted) == "write_briefing"


def test_parse_search_payload_json() -> None:
    raw = json.dumps(
        {
            "answer": "LangGraph orchestrates agents.",
            "sources": [{"url": "https://docs.langchain.com", "title": "Docs"}],
        }
    )
    answer, sources = parse_search_payload(raw)
    assert "LangGraph" in answer
    assert sources[0]["url"] == "https://docs.langchain.com"


class _FakeStructured:
    def __init__(self, factory) -> None:
        self.factory = factory

    def invoke(self, _messages: object) -> object:
        return self.factory()


class _FakeResearchLLM:
    def __init__(self, grades: list[EvidenceGrade], subquestions: list[str]) -> None:
        self._grades = list(grades)
        self._subquestions = subquestions
        self.grade_calls = 0
        self.draft_calls = 0

    def with_structured_output(self, schema: type, method: str = "json_schema") -> _FakeStructured:
        assert method == "json_schema"
        if schema is Subquestions:
            return _FakeStructured(
                lambda: Subquestions(subquestions=list(self._subquestions))
            )
        if schema is EvidenceGrade:
            def _next_grade() -> EvidenceGrade:
                self.grade_calls += 1
                if self._grades:
                    return self._grades.pop(0)
                return EvidenceGrade(relevant=True, reason="fallback", rewrite_query=None)

            return _FakeStructured(_next_grade)
        raise AssertionError(f"unexpected schema {schema}")

    def invoke(self, _messages: object, config: object = None, **_kwargs: object) -> object:
        self.draft_calls += 1
        return type("Resp", (), {"text": "Draft: LangGraph is a stateful agent runtime. https://example.com/docs", "content": ""})()


def _fake_search(queries: list[str]):
    def search(query: str) -> str:
        queries.append(query)
        return json.dumps(
            {
                "query": query,
                "answer": f"Results for {query}",
                "sources": [
                    {
                        "url": "https://example.com/docs",
                        "title": "Example docs",
                    }
                ],
            }
        )

    return search


def _fake_fetch(url: str) -> str:
    return f"Body of {url}"


def test_strong_evidence_skips_rewrite() -> None:
    queries: list[str] = []
    llm = _FakeResearchLLM(
        grades=[
            EvidenceGrade(relevant=True, reason="enough sources", rewrite_query=None),
        ],
        subquestions=["What is LangGraph?", "How does persistence work?"],
    )
    graph = build_research_graph(llm=llm, search=_fake_search(queries), fetch=_fake_fetch)
    result = graph.invoke(
        initial_research_state("What is LangGraph?"),
        config={"recursion_limit": RESEARCH_RECURSION_LIMIT},
    )
    assert result["loop_count"] == 1
    assert result["grade_relevant"] is True
    assert llm.grade_calls == 1
    assert llm.draft_calls == 1
    assert not any(note.startswith("Rewrote search query") for note in result["notes"])
    assert result["draft"].startswith("Draft:")
    assert any(source["url"] == "https://example.com/docs" for source in result["sources"])
    assert len(queries) == 1


def test_weak_evidence_rewrites_once() -> None:
    queries: list[str] = []
    llm = _FakeResearchLLM(
        grades=[
            EvidenceGrade(
                relevant=False,
                reason="too thin",
                rewrite_query="LangGraph checkpointer sqlite tutorial",
            ),
            EvidenceGrade(relevant=True, reason="now enough", rewrite_query=None),
        ],
        subquestions=["What is LangGraph?"],
    )
    graph = build_research_graph(llm=llm, search=_fake_search(queries), fetch=_fake_fetch)
    result = graph.invoke(
        initial_research_state("What is LangGraph?"),
        config={"recursion_limit": RESEARCH_RECURSION_LIMIT},
    )
    assert result["loop_count"] == 2
    assert result["grade_relevant"] is True
    assert llm.grade_calls == 2
    assert any("Rewrote search query to: LangGraph checkpointer sqlite tutorial" in n for n in result["notes"])
    assert queries[-1] == "LangGraph checkpointer sqlite tutorial"
    assert result["draft"]


def test_always_weak_stops_at_max_loops() -> None:
    queries: list[str] = []
    llm = _FakeResearchLLM(
        grades=[
            EvidenceGrade(relevant=False, reason="thin", rewrite_query="q2"),
            EvidenceGrade(relevant=False, reason="still thin", rewrite_query="q3"),
            EvidenceGrade(relevant=False, reason="should not run", rewrite_query="q4"),
        ],
        subquestions=["What is LangGraph?"],
    )
    graph = build_research_graph(llm=llm, search=_fake_search(queries), fetch=_fake_fetch)
    result = graph.invoke(
        initial_research_state("What is LangGraph?", max_loops=2),
        config={"recursion_limit": RESEARCH_RECURSION_LIMIT},
    )
    assert result["loop_count"] == 2
    assert result["grade_relevant"] is False
    assert llm.grade_calls == 2
    assert llm.draft_calls == 1
    assert "q3" not in queries
    assert queries[-1] == "q2"
    assert result["draft"]
