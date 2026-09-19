from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from briefing.checkpoint import open_sqlite_checkpointer
from briefing.graph import RESEARCH_RECURSION_LIMIT, build_research_graph
from briefing.nodes import after_human, parse_human_decision
from briefing.state import initial_research_state
from tests.test_research import (
    _FakeResearchLLM,
    _fake_fetch,
    _fake_search,
    _grounded_briefing,
)


def _config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RESEARCH_RECURSION_LIMIT,
    }


def test_parse_human_decision() -> None:
    assert parse_human_decision("a") == ("approve", "")
    assert parse_human_decision("more_research") == ("more_research", "")
    assert parse_human_decision({"action": "more_research", "query": "sqlite"}) == (
        "more_research",
        "sqlite",
    )


def test_after_human_routes() -> None:
    state = initial_research_state("q")
    state["human_decision"] = "more_research"
    assert after_human(state) == "rewrite"
    state["human_decision"] = "approve"
    assert after_human(state) == "__end__"


def test_interrupt_then_approve() -> None:
    llm = _FakeResearchLLM(
        grades=[],
        subquestions=["What is LangGraph?"],
    )
    graph = build_research_graph(
        llm=llm,
        search=_fake_search([]),
        fetch=_fake_fetch,
        checkpointer=InMemorySaver(),
    )
    config = _config("approve-thread")
    graph.invoke(initial_research_state("What is LangGraph?"), config)
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert snapshot.values["briefing"]["headline"] == _grounded_briefing().headline
    graph.invoke(Command(resume="approve"), config)
    done = graph.get_state(config)
    assert done.next == ()
    assert done.values["human_decision"] == "approve"


def test_more_research_searches_again() -> None:
    queries: list[str] = []
    llm = _FakeResearchLLM(
        grades=[],
        subquestions=["What is LangGraph?"],
    )
    graph = build_research_graph(
        llm=llm,
        search=_fake_search(queries),
        fetch=_fake_fetch,
        checkpointer=InMemorySaver(),
    )
    config = _config("more-thread")
    graph.invoke(initial_research_state("What is LangGraph?"), config)
    first_searches = list(queries)
    graph.invoke(
        Command(resume={"action": "more_research", "query": "LangGraph persistence"}),
        config,
    )
    snapshot = graph.get_state(config)
    assert snapshot.next == ("human_review",)
    assert "LangGraph persistence" in queries
    assert len(queries) > len(first_searches)
    assert any("Human requested more research" in note for note in snapshot.values["notes"])
    graph.invoke(Command(resume="approve"), config)
    assert graph.get_state(config).values["human_decision"] == "approve"


def test_sqlite_survives_new_graph_instance(tmp_path) -> None:
    db = tmp_path / "checkpoints.sqlite"
    saver = open_sqlite_checkpointer(db)
    llm = _FakeResearchLLM(grades=[], subquestions=["What is LangGraph?"])
    graph = build_research_graph(
        llm=llm,
        search=_fake_search([]),
        fetch=_fake_fetch,
        checkpointer=saver,
    )
    config = _config("persist-thread")
    graph.invoke(initial_research_state("What is LangGraph?"), config)
    assert graph.get_state(config).next == ("human_review",)

    saver2 = open_sqlite_checkpointer(db)
    graph2 = build_research_graph(
        llm=llm,
        search=_fake_search([]),
        fetch=_fake_fetch,
        checkpointer=saver2,
    )
    restored = graph2.get_state(config)
    assert restored.next == ("human_review",)
    assert restored.values["briefing"]["headline"] == "LangGraph"
    graph2.invoke(Command(resume="approve"), config)
    assert graph2.get_state(config).next == ()
    assert graph2.get_state(config).values["human_decision"] == "approve"
