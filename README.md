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

Run the research graph. It pauses for human approval and stores the thread in `data/checkpoints.sqlite` (gitignored):

```bash
uv run brief research "What is LangGraph used for?"
```

At the interrupt:

```
[a]pprove / [m]ore research / [q]uit
```

Quit leaves the thread paused. Resume or inspect it later:

```bash
uv run brief research --resume THREAD_ID
uv run brief replay THREAD_ID
```

The command prints markdown with a headline, summary, sourced findings, and a source list. A critic node rejects claims that cite URLs that were never collected.

## Graph

```
START → decompose → search → fetch → grade
                      ↑         │
                      │         ├─ weak evidence, loops left → rewrite ─┘
                      │         └─ else → write_briefing → critic
                      │                      │
                      │                      ├─ ungrounded, loops left → rewrite ─┘
                      │                      └─ else → human_review (interrupt)
                      │                                   ├─ more research → rewrite ─┘
                      └───────────────────────────────────┘
                                                          └─ approve → END
```

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

These tests do not call Gemini. `brief hello`, `brief summarize`, `brief ask`, and `brief research` do. Checkpoints live in `data/checkpoints.sqlite`.

## Roadmap

1. Scaffold + Gemini hello
2. URL summarizer with structured output
3. ReAct agent with search and fetch tools
4. Research graph with a grade-and-rewrite loop
5. Cited briefing + critic node
6. SQLite checkpoint + human approval (this commit)
