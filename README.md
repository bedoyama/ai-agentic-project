# Research briefing agent

Hands-on LangChain + LangGraph project powered by Gemini. A CLI that will grow from a hello-world call into a self-correcting research briefing agent.

## Setup

This repo reads a prepaid Gemini Developer API key from the environment. Prefer `GEMINI_API_KEY`. `GOOGLE_API_KEY` is a fallback.

If the key is already exported in your shell:

```bash
uv sync
uv run brief hello
```

You should see `Gemini is ready.` from the default model `gemini-3.8-flash`. The Google GenAI SDK may also print an automatic-function-calling notice to the terminal; that is SDK noise, not the model reply.

Summarize a URL into structured JSON (`title`, `url`, `key_points`, `caveats`):

```bash
uv run brief summarize https://docs.langchain.com/oss/python/langgraph/overview
```

Ask a current-events question. The ReAct graph streams each agent/tool step, then prints the answer:

```bash
uv run brief ask "Who is the current Python release manager?"
```

Run the research graph (decompose → search → fetch → grade, rewrite if evidence is weak, then a cited briefing + critic):

```bash
uv run brief research "What is LangGraph used for?"
```

The command prints markdown with a headline, summary, sourced findings, and a source list. A critic node rejects claims that cite URLs that were never collected.

To override the model:

```bash
export GEMINI_MODEL=gemini-3.7-flash
uv run brief hello
```

Optional local file (gitignored). Copy the example, then fill in values only if you do not already export them:

```bash
cp .env.example .env
```

## Tests

```bash
uv run pytest
```

These tests do not call Gemini. `brief hello`, `brief summarize`, `brief ask`, and `brief research` do.

## Roadmap

1. Scaffold + Gemini hello
2. URL summarizer with structured output
3. ReAct agent with search and fetch tools
4. Research graph with a grade-and-rewrite loop
5. Cited briefing + critic node (this commit)
6. SQLite checkpoint + human approval
