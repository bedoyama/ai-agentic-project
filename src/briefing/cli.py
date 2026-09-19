"""CLI for the research briefing agent."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import typer
from langchain.messages import HumanMessage
from langgraph.types import Command

from briefing.checkpoint import open_sqlite_checkpointer
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


def _research_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RESEARCH_RECURSION_LIMIT,
    }


def _print_briefing(state: dict) -> None:
    raw_briefing = state.get("briefing") or {}
    if raw_briefing:
        typer.echo("--- briefing ---")
        typer.echo(briefing_to_markdown(Briefing.model_validate(raw_briefing)))
    if "critic_grounded" in state:
        status = "accepted" if state.get("critic_grounded") else "rejected"
        typer.echo(f"Critic: {status}. {state.get('critic_reason', '')}".strip())


def _prompt_decision() -> str | dict | None:
    choice = typer.prompt("[a]pprove / [m]ore research / [q]uit", default="a").strip().lower()
    if choice.startswith("q"):
        return None
    if choice.startswith("m"):
        extra = typer.prompt(
            "Optional extra search query (blank to auto-pick)",
            default="",
        )
        if extra.strip():
            return {"action": "more_research", "query": extra.strip()}
        return "more_research"
    return "approve"


def _stream_research(graph: object, payload: object, config: dict) -> None:
    for item in graph.stream(  # type: ignore[attr-defined]
        payload,
        config,
        stream_mode=["updates", "values"],
    ):
        mode, data = item if isinstance(item, tuple) else ("updates", item)
        if mode == "updates" and isinstance(data, dict) and "__interrupt__" not in data:
            _print_research_update(data)


@app.command()
def research(
    question: str | None = typer.Argument(None, help="Research question"),
    resume: str | None = typer.Option(None, "--resume", help="Resume a paused thread"),
    thread_id: str | None = typer.Option(None, "--thread-id", help="Set the thread id"),
) -> None:
    """Run the research graph, pause for approval, and checkpoint to SQLite."""
    if resume:
        tid = resume
    elif question:
        tid = thread_id or str(uuid.uuid4())
    else:
        typer.secho("Provide a question or --resume THREAD_ID.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    try:
        graph = build_research_graph(checkpointer=open_sqlite_checkpointer())
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    config = _research_config(tid)
    typer.echo(f"Model: {get_model_name()}")
    typer.echo(f"Thread: {tid}")

    try:
        if not resume:
            _stream_research(graph, initial_research_state(question or ""), config)
        while True:
            snapshot = graph.get_state(config)
            if resume and not snapshot.values:
                typer.secho(f"No checkpoint for thread {tid}.", fg=typer.colors.RED, err=True)
                raise typer.Exit(code=1)
            if not snapshot.next:
                if snapshot.values:
                    _print_briefing(snapshot.values)
                    if snapshot.values.get("human_decision") == "approve":
                        typer.echo("Approved.")
                return
            _print_briefing(snapshot.values)
            decision = _prompt_decision()
            if decision is None:
                typer.echo(f"Paused. Resume with: uv run brief research --resume {tid}")
                return
            _stream_research(graph, Command(resume=decision), config)
            resume = None
    except Exception as exc:
        typer.secho(f"Failed to research: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def replay(thread_id: str) -> None:
    """Dump checkpointed state and history for a research thread."""
    try:
        graph = build_research_graph(checkpointer=open_sqlite_checkpointer())
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    config = _research_config(thread_id)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        typer.secho(f"No checkpoint for thread {thread_id}.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    values = snapshot.values
    typer.echo(f"Thread: {thread_id}")
    typer.echo(f"Next: {snapshot.next or ('END',)}")
    checkpoint_id = (snapshot.config or {}).get("configurable", {}).get("checkpoint_id")
    typer.echo(f"Checkpoint: {checkpoint_id}")
    typer.echo(f"Question: {values.get('question')}")
    typer.echo(f"Loops: {values.get('loop_count')}/{values.get('max_loops')}")
    typer.echo(f"Human: {values.get('human_decision') or '(waiting)'}")
    _print_briefing(values)
    typer.echo("--- history ---")
    for index, hist in enumerate(graph.get_state_history(config)):
        if index >= 20:
            typer.echo("...")
            break
        nxt = hist.next or ("END",)
        hist_id = (hist.config or {}).get("configurable", {}).get("checkpoint_id")
        typer.echo(f"{index}: next={nxt} id={hist_id}")


if __name__ == "__main__":
    app()
