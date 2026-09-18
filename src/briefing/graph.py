"""LangGraph graphs: ReAct ask agent and the research briefing loop."""

from __future__ import annotations

from typing import Any

from langchain.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from briefing.config import get_llm
from briefing.nodes import after_grade, make_research_nodes
from briefing.state import ResearchState
from briefing.tools import fetch_page, fetch_url, run_web_search, web_search

ASK_SYSTEM = (
    "You are a research assistant. Use web_search for current or factual questions. "
    "Use fetch_url when you have a specific page to read. "
    "After tools return, write a short answer and include source URLs. "
    "Do not invent URLs; only cite sources the tools provided."
)

ASK_RECURSION_LIMIT = 8


def build_ask_graph(
    *,
    llm: Any = None,
    tools: list[BaseTool] | None = None,
) -> Any:
    """Compile a MessagesState graph: agent ⇄ tools until the model stops calling tools."""
    tool_list = tools or [web_search, fetch_url]
    model = (llm or get_llm()).bind_tools(tool_list)

    def agent(state: MessagesState) -> dict:
        messages = list(state["messages"])
        if not messages or getattr(messages[0], "type", None) != "system":
            messages = [SystemMessage(content=ASK_SYSTEM), *messages]
        response = model.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tool_list))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


RESEARCH_RECURSION_LIMIT = 16


def build_research_graph(
    *,
    llm: Any = None,
    search: Any = run_web_search,
    fetch: Any = fetch_page,
) -> Any:
    """Compile decompose → search → fetch → grade ⇄ rewrite, then a draft briefing."""
    nodes = make_research_nodes(llm=llm or get_llm(), search=search, fetch=fetch)
    graph = StateGraph(ResearchState)
    graph.add_node("decompose", nodes["decompose"])
    graph.add_node("search", nodes["search"])
    graph.add_node("fetch", nodes["fetch"])
    graph.add_node("grade", nodes["grade"])
    graph.add_node("rewrite", nodes["rewrite"])
    graph.add_node("write_briefing", nodes["write_briefing"])
    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "search")
    graph.add_edge("search", "fetch")
    graph.add_edge("fetch", "grade")
    graph.add_conditional_edges(
        "grade",
        after_grade,
        {"rewrite": "rewrite", "write_briefing": "write_briefing"},
    )
    graph.add_edge("rewrite", "search")
    graph.add_edge("write_briefing", END)
    return graph.compile()
