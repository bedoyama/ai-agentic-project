"""CLI for the research briefing agent."""

from __future__ import annotations

from collections.abc import Callable

import typer
from langchain.messages import HumanMessage

from briefing.config import ConfigError, get_llm, get_model_name
from briefing.graph import ASK_RECURSION_LIMIT, build_ask_graph
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


if __name__ == "__main__":
    app()
