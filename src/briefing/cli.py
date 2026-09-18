"""CLI for the research briefing agent."""

from __future__ import annotations

import typer
from langchain.messages import HumanMessage

from briefing.config import ConfigError, get_llm, get_model_name

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


if __name__ == "__main__":
    app()
