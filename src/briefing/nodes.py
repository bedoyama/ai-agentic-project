"""Research-graph node functions (decompose, search, fetch, grade, rewrite, draft)."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain.messages import HumanMessage, SystemMessage

from briefing.schemas import EvidenceGrade, Subquestions
from briefing.state import DEFAULT_MAX_LOOPS, ResearchState

MAX_SUBQUESTIONS = 3
MAX_FETCH_URLS = 2
MAX_FETCH_CHARS = 1500
MAX_GRADE_CHARS = 8000
_URL_RE = re.compile(r"https?://[^\s)\]>'\"}]+")


def _as_model(result: Any, schema: type) -> Any:
    if isinstance(result, schema):
        return result
    return schema.model_validate(result)


def _structured(llm: Any, schema: type) -> Any:
    return llm.with_structured_output(schema, method="json_schema")


def parse_search_payload(raw: str) -> tuple[str, list[dict[str, Any]]]:
    """Turn web_search JSON (or loose text) into an answer plus source dicts."""
    answer = ""
    sources: list[dict[str, Any]] = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        answer = str(data.get("answer") or "").strip()
        for item in data.get("sources") or []:
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if not url:
                continue
            sources.append(
                {
                    "url": url,
                    "title": item.get("title") or "",
                    "snippet": (item.get("snippet") or answer)[:400],
                }
            )
    else:
        answer = raw.strip()

    if not sources:
        for url in _URL_RE.findall(answer):
            sources.append({"url": url.rstrip(".,;"), "title": "", "snippet": answer[:400]})
    return answer, sources


def after_grade(state: ResearchState) -> Literal["rewrite", "write_briefing"]:
    """Loop back to search only while evidence is weak and under the cap."""
    max_loops = state.get("max_loops") or DEFAULT_MAX_LOOPS
    if not state.get("grade_relevant") and state.get("loop_count", 0) < max_loops:
        return "rewrite"
    return "write_briefing"


def make_research_nodes(
    *,
    llm: Any,
    search: Any,
    fetch: Any,
) -> dict[str, Any]:
    def decompose(state: ResearchState) -> dict[str, Any]:
        result = _structured(llm, Subquestions).invoke(
            [
                SystemMessage(
                    content=(
                        "Split the user question into 1-3 focused web-search sub-questions. "
                        "Do not answer the question."
                    )
                ),
                HumanMessage(content=state["question"]),
            ]
        )
        parsed = _as_model(result, Subquestions)
        subquestions = [q.strip() for q in parsed.subquestions if q.strip()][:MAX_SUBQUESTIONS]
        if not subquestions:
            subquestions = [state["question"]]
        query = " | ".join(subquestions)
        return {"subquestions": subquestions, "search_query": query}

    def search_node(state: ResearchState) -> dict[str, Any]:
        query = state.get("search_query") or state["question"]
        raw = search(query)
        answer, sources = parse_search_payload(raw)
        note = f"Search [{query}]: {answer}" if answer else f"Search [{query}]: no answer text"
        urls = [source["url"] for source in sources][:MAX_FETCH_URLS]
        return {
            "sources": sources,
            "notes": [note],
            "latest_urls": urls,
        }

    def fetch_node(state: ResearchState) -> dict[str, Any]:
        urls = (state.get("latest_urls") or [])[:MAX_FETCH_URLS]
        if not urls:
            return {"notes": ["No URLs to fetch."]}
        notes: list[str] = []
        for url in urls:
            try:
                text = fetch(url)
                notes.append(f"Fetched {url}: {text[:MAX_FETCH_CHARS]}")
            except Exception as exc:
                notes.append(f"Failed to fetch {url}: {exc}")
        return {"notes": notes}

    def grade(state: ResearchState) -> dict[str, Any]:
        evidence = "\n".join(state.get("notes") or [])[:MAX_GRADE_CHARS]
        source_lines = "\n".join(
            f"- {s.get('title') or ''} {s.get('url')}" for s in state.get("sources") or []
        )
        result = _structured(llm, EvidenceGrade).invoke(
            [
                SystemMessage(
                    content=(
                        "Grade whether the evidence is enough to draft a grounded briefing. "
                        "If it is not, set relevant=false and provide rewrite_query."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n"
                        f"Subquestions: {state.get('subquestions')}\n"
                        f"Sources:\n{source_lines or '(none)'}\n\n"
                        f"Notes:\n{evidence or '(none)'}"
                    )
                ),
            ]
        )
        parsed = _as_model(result, EvidenceGrade)
        return {
            "grade_relevant": parsed.relevant,
            "grade_reason": parsed.reason,
            "rewrite_query": parsed.rewrite_query or "",
            "loop_count": state.get("loop_count", 0) + 1,
        }

    def rewrite(state: ResearchState) -> dict[str, Any]:
        query = (state.get("rewrite_query") or "").strip() or state["question"]
        return {
            "search_query": query,
            "notes": [f"Rewrote search query to: {query}"],
        }

    def write_briefing(state: ResearchState) -> dict[str, Any]:
        source_lines = "\n".join(
            f"- {s.get('title') or ''} {s.get('url')}" for s in state.get("sources") or []
        )
        notes = "\n".join(state.get("notes") or [])[:MAX_GRADE_CHARS]
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Write a one-paragraph draft briefing. Use only the notes and sources. "
                        "Mention source URLs. If evidence is weak, say what is still unknown."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n"
                        f"Grade: relevant={state.get('grade_relevant')} "
                        f"({state.get('grade_reason')})\n"
                        f"Sources:\n{source_lines or '(none)'}\n\n"
                        f"Notes:\n{notes or '(none)'}"
                    )
                ),
            ]
        )
        text = getattr(response, "text", None)
        if not (isinstance(text, str) and text.strip()):
            content = getattr(response, "content", "")
            text = content if isinstance(content, str) else str(content)
        return {"draft": text.strip()}

    return {
        "decompose": decompose,
        "search": search_node,
        "fetch": fetch_node,
        "grade": grade,
        "rewrite": rewrite,
        "write_briefing": write_briefing,
    }
