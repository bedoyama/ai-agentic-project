"""Research-graph node functions (decompose, search, fetch, grade, rewrite, briefing, critic)."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain.messages import HumanMessage, SystemMessage

from langgraph.graph import END

from briefing.schemas import Briefing, CriticReport, EvidenceGrade, SourceRef, Subquestions
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


def allowed_source_urls(sources: list[dict[str, Any]] | None) -> set[str]:
    return {source.get("url", "") for source in sources or [] if source.get("url")}


def unsupported_source_claims(
    briefing: Briefing,
    sources: list[dict[str, Any]] | None,
) -> list[str]:
    """Flag findings that cite no URL, or a URL that was never collected."""
    allowed = allowed_source_urls(sources)
    issues: list[str] = []
    for finding in briefing.findings:
        if not finding.source_urls:
            issues.append(f"Unsourced claim: {finding.claim}")
            continue
        unknown = [url for url in finding.source_urls if url not in allowed]
        if unknown:
            issues.append(
                f"Unknown URL {unknown} for claim: {finding.claim}"
            )
    return issues


def briefing_to_markdown(briefing: Briefing) -> str:
    lines = [f"# {briefing.headline}", "", briefing.summary]
    if briefing.findings:
        lines.extend(["", "## Findings"])
        for finding in briefing.findings:
            cites = ", ".join(finding.source_urls) if finding.source_urls else "no source"
            lines.append(f"- {finding.claim} ({finding.confidence}; {cites})")
    if briefing.open_questions:
        lines.extend(["", "## Open questions"])
        for question in briefing.open_questions:
            lines.append(f"- {question}")
    if briefing.sources:
        lines.extend(["", "## Sources"])
        for source in briefing.sources:
            title = source.title or source.url
            lines.append(f"- [{title}]({source.url})")
    return "\n".join(lines).strip() + "\n"


def after_critic(state: ResearchState) -> Literal["rewrite", "__end__"]:
    """Send an ungrounded briefing back to search if loops remain."""
    max_loops = state.get("max_loops") or DEFAULT_MAX_LOOPS
    if not state.get("critic_grounded") and state.get("loop_count", 0) < max_loops:
        return "rewrite"
    return END


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
        catalog = [
            SourceRef(url=source["url"], title=source.get("title") or "")
            for source in state.get("sources") or []
            if source.get("url")
        ]
        source_lines = "\n".join(f"- {item.title} {item.url}".strip() for item in catalog)
        notes = "\n".join(state.get("notes") or [])[:MAX_GRADE_CHARS]
        result = _structured(llm, Briefing).invoke(
            [
                SystemMessage(
                    content=(
                        "Write a cited research briefing. "
                        "Every finding.claim must have source_urls taken only from the "
                        "provided source list. Do not invent URLs. "
                        "If evidence is weak, put remaining gaps in open_questions "
                        "instead of unsourced claims."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {state['question']}\n"
                        f"Grade: relevant={state.get('grade_relevant')} "
                        f"({state.get('grade_reason')})\n"
                        f"Allowed sources:\n{source_lines or '(none)'}\n\n"
                        f"Notes:\n{notes or '(none)'}"
                    )
                ),
            ]
        )
        parsed = _as_model(result, Briefing).model_copy(update={"sources": catalog})
        return {"briefing": parsed.model_dump()}

    def critic(state: ResearchState) -> dict[str, Any]:
        raw = state.get("briefing") or {}
        if not raw:
            return {
                "critic_grounded": False,
                "critic_reason": "No briefing to review.",
                "rewrite_query": state["question"],
                "notes": ["Critic rejected briefing: missing briefing."],
            }
        briefing = _as_model(raw, Briefing)
        issues = unsupported_source_claims(briefing, state.get("sources") or [])
        if not issues:
            notes = "\n".join(state.get("notes") or [])[:MAX_GRADE_CHARS]
            report = _as_model(
                _structured(llm, CriticReport).invoke(
                    [
                        SystemMessage(
                            content=(
                                "Check that each finding is actually supported by the notes "
                                "and the cited URLs. Set grounded=false if claims are invented "
                                "or weakly supported. Only use rewrite_query when grounded is false."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"Question: {state['question']}\n"
                                f"Briefing:\n{briefing_to_markdown(briefing)}\n"
                                f"Notes:\n{notes or '(none)'}"
                            )
                        ),
                    ]
                ),
                CriticReport,
            )
            if report.grounded and not report.unsupported_claims:
                return {
                    "critic_grounded": True,
                    "critic_reason": report.reason,
                }
            issues = report.unsupported_claims or [report.reason]
            rewrite_query = report.rewrite_query or f"primary sources for: {state['question']}"
        else:
            rewrite_query = f"primary sources for: {state['question']}"
        reason = "; ".join(issues)
        return {
            "critic_grounded": False,
            "critic_reason": reason,
            "rewrite_query": rewrite_query,
            "notes": [f"Critic rejected briefing: {reason}"],
        }

    return {
        "decompose": decompose,
        "search": search_node,
        "fetch": fetch_node,
        "grade": grade,
        "rewrite": rewrite,
        "write_briefing": write_briefing,
        "critic": critic,
    }
