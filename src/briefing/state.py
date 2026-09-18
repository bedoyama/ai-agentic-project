"""Custom LangGraph state for the research briefing loop."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

DEFAULT_MAX_LOOPS = 2


def merge_sources(
    existing: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Append sources, keyed by URL so later rounds update rather than duplicate."""
    merged: dict[str, dict[str, Any]] = {}
    for source in existing or []:
        url = source.get("url")
        if url:
            merged[url] = source
    for source in new or []:
        url = source.get("url")
        if not url:
            continue
        merged[url] = {**merged.get(url, {}), **source}
    return list(merged.values())


class ResearchState(TypedDict):
    question: str
    subquestions: list[str]
    search_query: str
    sources: Annotated[list[dict[str, Any]], merge_sources]
    notes: Annotated[list[str], operator.add]
    latest_urls: list[str]
    grade_relevant: bool
    grade_reason: str
    rewrite_query: str
    loop_count: int
    max_loops: int
    draft: str


def initial_research_state(
    question: str,
    *,
    max_loops: int = DEFAULT_MAX_LOOPS,
) -> ResearchState:
    return {
        "question": question,
        "subquestions": [],
        "search_query": question,
        "sources": [],
        "notes": [],
        "latest_urls": [],
        "grade_relevant": False,
        "grade_reason": "",
        "rewrite_query": "",
        "loop_count": 0,
        "max_loops": max_loops,
        "draft": "",
    }
