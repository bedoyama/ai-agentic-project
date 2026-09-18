"""Summarize a URL into a structured UrlSummary."""

from __future__ import annotations

from collections.abc import Callable

from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from briefing.config import get_llm
from briefing.schemas import UrlSummary
from briefing.tools import fetch_page

_SYSTEM = (
    "You summarize web pages for a research briefing. "
    "Use only the provided page content. "
    "If the content is thin, say so in caveats instead of inventing facts. "
    "key_points must be concrete and specific."
)


def _as_summary(result: UrlSummary | dict, url: str) -> UrlSummary:
    summary = result if isinstance(result, UrlSummary) else UrlSummary.model_validate(result)
    return summary.model_copy(update={"url": url})


def summarize_url(
    url: str,
    *,
    fetch: Callable[[str], str] = fetch_page,
    llm: ChatGoogleGenerativeAI | None = None,
) -> UrlSummary:
    """Fetch a URL and return a structured summary."""
    content = fetch(url)
    model = llm or get_llm()
    structured = model.with_structured_output(UrlSummary, method="json_schema")
    result = structured.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"URL: {url}\n\nPAGE CONTENT:\n{content}"),
        ]
    )
    return _as_summary(result, url)
