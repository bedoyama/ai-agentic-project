"""Pydantic models for structured Gemini output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UrlSummary(BaseModel):
    """Structured summary of a single web page."""

    title: str = Field(description="Page title")
    url: str = Field(description="Canonical URL that was summarized")
    key_points: list[str] = Field(
        description="3-7 factual takeaways taken only from the page content"
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Limitations, unknowns, or warnings stated on the page. Empty if none.",
    )
