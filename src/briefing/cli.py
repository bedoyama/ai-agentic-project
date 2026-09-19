"""CLI for the research briefing agent."""

from __future__ import annotations

from collections.abc import Callable

import typer
from langchain.messages import HumanMessage

from briefing.config import ConfigError, get_llm, get_model_name
from briefing.graph import (
    ASK_RECURSION_LIMIT,
    RESEARCH_RECURSION_LIMIT,
    build_ask_graph,
    build_research_graph,
)
from briefing.nodes import briefing_to_markdown
from briefing.schemas import Briefing
from briefing.state import initial_research_state
from briefing.summarize import summarize_url

app = typer.Typer(
    no_args_is_help=True,
    help="Research briefing agent (LangChain + LangGraph + Gemini).",
)


@app.callback()
def main() -> None:
    """Research briefing agent (LangChain + LangGraph + Gemini)."""


@app.command()
def hello() -> None:
    """Send a tiny prompt to Gemini to verify the API key and model."""
    try:
        llm = get_llm()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Model: {get_model_name()}")
    response = llm.invoke(
        [HumanMessage(content="Reply with exactly: Gemini is ready.")]
    )
    text = response.text if hasattr(response, "text") else str(response.content)
    typer.echo(text)


@app.command()
def summarize(url: str) -> None:
    """Fetch a URL and print a structured JSON summary."""
    try:
        summary = summarize_url(url)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.secho(f"Failed to summarize URL: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(summary.model_dump_json(indent=2))


def _message_text(message: object) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list) and not content:
        return ""
    return str(content) if content else ""


def _preview(text: str, limit: int = 400) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit] + "..."


def _print_ask_update(
    update: dict,
    echo: Callable[[str], None] = typer.echo,
) -> None:
    for node, payload in update.items():
        echo(f"--- {node} ---")
        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not messages:
            echo(str(payload))
            continue
        for message in messages:
            kind = getattr(message, "type", message.__class__.__name__)
            tool_calls = getattr(message, "tool_calls", None) or []
            if kind == "ai" and tool_calls:
                for call in tool_calls:
                    echo(f"tool call: {call['name']} {call.get('args', {})}")
            text = _message_text(message)
            if kind == "tool":
                echo(f"{getattr(message, 'name', 'tool')}: {_preview(text)}")
            elif text:
                echo(text)


@app.command()
def ask(question: str) -> None:
    """Answer a question with a LangGraph ReAct loop (web_search + fetch_url)."""
    try:
        graph = build_ask_graph()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Model: {get_model_name()}")
    final_text = ""
    try:
        for update in graph.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="updates",
            config={"recursion_limit": ASK_RECURSION_LIMIT},
        ):
            _print_ask_update(update)
            agent_payload = update.get("agent") if isinstance(update, dict) else None
            if not agent_payload:
                continue
            for message in agent_payload.get("messages", []):
                if getattr(message, "tool_calls", None):
                    continue
                text = _message_text(message)
                if text:
                    final_text = text
    except Exception as exc:
        typer.secho(f"Failed to answer: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    if final_text:
        typer.echo("--- answer ---")
        typer.echo(final_text)


def _print_research_update(
    update: dict,
    echo: Callable[[str], None] = typer.echo,
) -> None:
    for node, payload in update.items():
        echo(f"--- {node} ---")
        if not isinstance(payload, dict):
            echo(str(payload))
            continue
        if payload.get("subquestions"):
            echo("subquestions:")
            for item in payload["subquestions"]:
                echo(f"- {item}")
        if payload.get("search_query"):
            echo(f"search_query: {payload['search_query']}")
        if payload.get("latest_urls"):
            echo("urls: " + ", ".join(payload["latest_urls"]))
        if "grade_relevant" in payload:
            echo(f"relevant: {payload['grade_relevant']}")
            echo(f"reason: {payload.get('grade_reason', '')}")
            echo(f"loop: {payload.get('loop_count')}")
        if payload.get("rewrite_query"):
            echo(f"rewrite_query: {payload['rewrite_query']}")
        notes = payload.get("notes") or []
        if notes:
            echo(_preview(notes[-1]))
        briefing = payload.get("briefing") or {}
        if briefing.get("headline"):
            echo(f"headline: {briefing['headline']}")
        if "critic_grounded" in payload:
            echo(f"grounded: {payload['critic_grounded']}")
            if payload.get("critic_reason"):
                echo(payload["critic_reason"])


@app.command()
def research(question: str) -> None:
    """Run the research graph and print a cited markdown briefing."""
    try:
        graph = build_research_graph()
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Model: {get_model_name()}")
    final: dict = {}
    try:
        for item in graph.stream(
            initial_research_state(question),
            stream_mode=["updates", "values"],
            config={"recursion_limit": RESEARCH_RECURSION_LIMIT},
        ):
            mode, data = item if isinstance(item, tuple) else ("updates", item)
            if mode == "updates":
                _print_research_update(data)
            elif mode == "values" and isinstance(data, dict):
                final = data
    except Exception as exc:
        typer.secho(f"Failed to research: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("--- briefing ---")
    raw_briefing = final.get("briefing") or {}
    if raw_briefing:
        typer.echo(briefing_to_markdown(Briefing.model_validate(raw_briefing)))
    if "critic_grounded" in final:
        status = "accepted" if final.get("critic_grounded") else "rejected"
        typer.echo(f"Critic: {status}. {final.get('critic_reason', '')}".strip())


if __name__ == "__main__":
    app()
