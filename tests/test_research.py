from __future__ import annotations

import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from briefing.graph import RESEARCH_RECURSION_LIMIT, build_research_graph
from briefing.nodes import (
    after_critic,
    after_grade,
    briefing_to_markdown,
    make_research_nodes,
    parse_search_payload,
    unsupported_source_claims,
)
from briefing.schemas import Briefing, CriticReport, EvidenceGrade, Finding, SourceRef, Subquestions
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


def _grounded_briefing() -> Briefing:
    return Briefing(
        headline="LangGraph",
        summary="LangGraph is a stateful agent runtime.",
        findings=[
            Finding(
                claim="LangGraph is a stateful agent runtime.",
                source_urls=["https://example.com/docs"],
                confidence="high",
            )
        ],
        open_questions=[],
        sources=[SourceRef(url="https://example.com/docs", title="Example docs")],
    )


class _FakeResearchLLM:
    def __init__(
        self,
        grades: list[EvidenceGrade],
        subquestions: list[str],
        *,
        briefing: Briefing | None = None,
        critic: CriticReport | None = None,
    ) -> None:
        self._grades = list(grades)
        self._subquestions = subquestions
        self._briefing = briefing or _grounded_briefing()
        self._critic = critic or CriticReport(
            grounded=True, unsupported_claims=[], reason="claims match sources"
        )
        self.grade_calls = 0
        self.briefing_calls = 0
        self.critic_calls = 0

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
        if schema is Briefing:
            def _briefing() -> Briefing:
                self.briefing_calls += 1
                return self._briefing

            return _FakeStructured(_briefing)
        if schema is CriticReport:
            def _critic() -> CriticReport:
                self.critic_calls += 1
                return self._critic

            return _FakeStructured(_critic)
        raise AssertionError(f"unexpected schema {schema}")


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


def _finish_research(llm: _FakeResearchLLM, search, fetch, state: dict) -> dict:
    graph = build_research_graph(
        llm=llm, search=search, fetch=fetch, checkpointer=InMemorySaver()
    )
    config = {
        "configurable": {"thread_id": "test"},
        "recursion_limit": RESEARCH_RECURSION_LIMIT,
    }
    graph.invoke(state, config)
    graph.invoke(Command(resume="approve"), config)
    return graph.get_state(config).values


def test_strong_evidence_skips_rewrite() -> None:
    queries: list[str] = []
    llm = _FakeResearchLLM(
        grades=[
            EvidenceGrade(relevant=True, reason="enough sources", rewrite_query=None),
        ],
        subquestions=["What is LangGraph?", "How does persistence work?"],
    )
    result = _finish_research(
        llm, _fake_search(queries), _fake_fetch, initial_research_state("What is LangGraph?")
    )
    assert result["loop_count"] == 1
    assert result["grade_relevant"] is True
    assert llm.grade_calls == 1
    assert llm.briefing_calls == 1
    assert llm.critic_calls == 1
    assert result["critic_grounded"] is True
    assert not any(note.startswith("Rewrote search query") for note in result["notes"])
    assert result["briefing"]["headline"] == "LangGraph"
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
    result = _finish_research(
        llm, _fake_search(queries), _fake_fetch, initial_research_state("What is LangGraph?")
    )
    assert result["loop_count"] == 2
    assert result["grade_relevant"] is True
    assert llm.grade_calls == 2
    assert any("Rewrote search query to: LangGraph checkpointer sqlite tutorial" in n for n in result["notes"])
    assert queries[-1] == "LangGraph checkpointer sqlite tutorial"
    assert result["briefing"]["headline"] == "LangGraph"
    assert result["critic_grounded"] is True


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
    result = _finish_research(
        llm,
        _fake_search(queries),
        _fake_fetch,
        initial_research_state("What is LangGraph?", max_loops=2),
    )
    assert result["loop_count"] == 2
    assert result["grade_relevant"] is False
    assert llm.grade_calls == 2
    assert llm.briefing_calls == 1
    assert llm.critic_calls == 1
    assert "q3" not in queries
    assert queries[-1] == "q2"
    assert result["briefing"]["headline"] == "LangGraph"


def test_unsupported_source_claims_flags_unknown_url() -> None:
    briefing = Briefing(
        headline="Bad",
        summary="Invented",
        findings=[
            Finding(
                claim="LangGraph runs on Mars.",
                source_urls=["https://evil.example/made-up"],
                confidence="high",
            )
        ],
    )
    issues = unsupported_source_claims(
        briefing, [{"url": "https://example.com/docs", "title": "Docs"}]
    )
    assert issues
    assert "https://evil.example/made-up" in issues[0]


def test_critic_rejects_url_not_in_sources() -> None:
    llm = _FakeResearchLLM(grades=[], subquestions=["q"])
    nodes = make_research_nodes(llm=llm, search=lambda _q: "", fetch=lambda _u: "")
    state = initial_research_state("What is LangGraph?")
    state["sources"] = [{"url": "https://example.com/docs", "title": "Docs"}]
    state["loop_count"] = 1
    state["briefing"] = Briefing(
        headline="Bad",
        summary="Invented",
        findings=[
            Finding(
                claim="LangGraph runs on Mars.",
                source_urls=["https://evil.example/made-up"],
                confidence="high",
            )
        ],
    ).model_dump()
    update = nodes["critic"](state)
    assert update["critic_grounded"] is False
    assert "Unknown URL" in update["critic_reason"]
    assert llm.critic_calls == 0
    merged = {**state, **update}
    assert after_critic(merged) == "rewrite"


def test_after_critic_pauses_for_human_when_loops_exhausted() -> None:
    state = initial_research_state("q", max_loops=2)
    state["critic_grounded"] = False
    state["loop_count"] = 2
    assert after_critic(state) == "human_review"


def test_briefing_markdown_includes_sources() -> None:
    markdown = briefing_to_markdown(_grounded_briefing())
    assert markdown.startswith("# LangGraph")
    assert "https://example.com/docs" in markdown
    assert "## Sources" in markdown
