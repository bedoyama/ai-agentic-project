"""LangGraph ReAct agent for question answering with search and fetch tools."""

from __future__ import annotations

from typing import Any

from langchain.messages import SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from briefing.config import get_llm
from briefing.tools import fetch_url, web_search

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
