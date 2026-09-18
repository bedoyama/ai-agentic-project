"""LangChain tools for web search and fetching page content."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import httpx
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from briefing.config import get_llm

_URL_RE = re.compile(r"https?://[^\s)\]>'\"}]+")

MAX_PAGE_CHARS = 24_000
_HTTP_HEADERS = {
    "User-Agent": "briefing-agent/0.1 (educational LangChain project)",
    "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
}


class _HTMLTextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = " ".join(data.split())
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return "\n".join(self._chunks)


def html_to_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def _looks_useful(text: str | None) -> bool:
    return bool(text and len(text.strip()) >= 40)


def _fetch_via_url_context(url: str, llm: ChatGoogleGenerativeAI) -> str | None:
    try:
        bound = llm.bind_tools([{"url_context": {}}])
        response = bound.invoke(
            "Read this URL and return the page title and the main body text only. "
            "Do not add commentary or a summary.\n\n"
            f"{url}"
        )
    except Exception:
        return None
    text = (response.text or "").strip()
    return text if _looks_useful(text) else None


def _fetch_via_http(url: str) -> str:
    with httpx.Client(
        follow_redirects=True,
        timeout=20.0,
        headers=_HTTP_HEADERS,
    ) as client:
        response = client.get(url)
        response.raise_for_status()
    body = response.text
    content_type = response.headers.get("content-type", "").lower()
    if "html" in content_type or body.lstrip().startswith("<"):
        body = html_to_text(body)
    body = body.strip()
    if not body:
        raise RuntimeError(f"No readable content at {url}")
    return body[:MAX_PAGE_CHARS]


def fetch_page(url: str, *, llm: ChatGoogleGenerativeAI | None = None) -> str:
    """Fetch readable page text via Gemini url_context, then HTTP if needed."""
    model = llm or get_llm()
    extracted = _fetch_via_url_context(url, model)
    if extracted:
        return extracted[:MAX_PAGE_CHARS]
    return _fetch_via_http(url)


@tool
def fetch_url_context(url: str) -> str:
    """Fetch readable content from a URL.

    Uses Gemini url_context when available, and falls back to a plain HTTP GET.
    Return the page title and main body text, not a summary.
    """
    return fetch_page(url)


@tool
def fetch_url(url: str) -> str:
    """Open a specific URL and return its readable title and body text."""
    return fetch_page(url)


def extract_search_sources(response: object) -> list[dict[str, str]]:
    """Pull citation URLs out of a Gemini google_search response."""
    sources: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(url: str | None, title: str | None = None) -> None:
        if not url:
            return
        cleaned = url.rstrip(".,;")
        if cleaned in seen:
            return
        seen.add(cleaned)
        item = {"url": cleaned}
        if title:
            item["title"] = title
        sources.append(item)

    blocks = getattr(response, "content_blocks", None) or []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for annotation in block.get("annotations") or []:
            if not isinstance(annotation, dict):
                continue
            extras = annotation.get("extras") or {}
            url = annotation.get("url") or extras.get("url")
            title = annotation.get("title") or extras.get("title")
            add(url, title)

    metadata = getattr(response, "response_metadata", None) or {}
    grounding = (
        metadata.get("grounding_metadata")
        or metadata.get("groundingMetadata")
        or {}
    )
    chunks = grounding.get("grounding_chunks") or grounding.get("groundingChunks") or []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        web = chunk.get("web") or {}
        add(web.get("uri") or web.get("url"), web.get("title"))

    if not sources:
        text = getattr(response, "text", "") or ""
        for match in _URL_RE.findall(text):
            add(match)

    return sources


def run_web_search(query: str, *, llm: ChatGoogleGenerativeAI | None = None) -> str:
    """Search the web with Gemini google_search and return JSON (answer + sources)."""
    model = (llm or get_llm()).bind_tools([{"google_search": {}}])
    response = model.invoke(
        "Search the public web for this query. Return a concise factual brief "
        "and keep source URLs. Query:\n"
        f"{query}"
    )
    payload = {
        "query": query,
        "answer": (response.text or "").strip(),
        "sources": extract_search_sources(response),
    }
    return json.dumps(payload, indent=2)


@tool
def web_search(query: str) -> str:
    """Search the public web for current facts, news, or documentation.

    Returns JSON with keys query, answer, and sources (url/title).
    Use this before answering questions that need up-to-date information.
    """
    return run_web_search(query)
