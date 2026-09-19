# Research briefing agent

A hands-on LangChain + LangGraph CLI that turns a question into a cited briefing. It decomposes the question, searches the web with Gemini, grades evidence, rewrites the query when the evidence is weak, drafts a structured briefing, runs a critic, then pauses for you to approve or request more research.

This is a learning project, not production software.

## Setup

You need Python 3.12+ and [uv](https://docs.astral.sh/uv/). Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) and export it:

```bash
export GEMINI_API_KEY=your-key
```

`GOOGLE_API_KEY` is accepted as a fallback. Then:

```bash
uv sync
uv run brief hello
```

You should see `Gemini is ready.` from the default model `gemini-3.8-flash`. The Google GenAI SDK may also print an automatic-function-calling notice; that is SDK noise, not the model reply.

Optional local file (gitignored):

```bash
cp .env.example .env
```

Override the model with `export GEMINI_MODEL=gemini-3.7-flash` (or whatever AI Studio lists for your key).

## Commands

| Command | What it does |
|---|---|
| `uv run brief hello` | Tiny Gemini ping |
| `uv run brief summarize <url>` | Structured JSON summary of a page |
| `uv run brief ask "..."` | ReAct loop with `web_search` and `fetch_url` |
| `uv run brief research "..."` | Full research graph, then human approval |
| `uv run brief research --resume THREAD_ID` | Resume a paused research thread |
| `uv run brief replay THREAD_ID` | Dump checkpointed state and history |

Research threads are stored in `data/checkpoints.sqlite` (gitignored). At the interrupt:

```
[a]pprove / [m]ore research / [q]uit
```

Quit leaves the thread paused. Gemini Google Search citations often use grounding-redirect URLs; the source list titles are the readable names.

Save local run output under `.out/` if you want; that directory is gitignored.

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

## How it was built

The git history is the curriculum. Read the commits in order:

1. Scaffold + Gemini hello
2. URL summarizer with structured output
3. ReAct agent with search and fetch tools
4. Research graph with a grade-and-rewrite loop
5. Cited briefing + critic node
6. SQLite checkpoint + human approval

## Tests

```bash
uv run pytest
```

These tests do not call Gemini. The `brief` commands above do.

## License

MIT. See [LICENSE](LICENSE).
