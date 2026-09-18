"""Pydantic models for structured Gemini output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Subquestions(BaseModel):
    """Decomposed research questions."""

    subquestions: list[str] = Field(
        description="1 to 3 focused sub-questions that would answer the user's question"
    )


class EvidenceGrade(BaseModel):
    """Whether current notes and sources are enough to draft a briefing."""

    relevant: bool = Field(
        description="True if the evidence is enough to draft a grounded briefing"
    )
    reason: str = Field(description="Short explanation of the grade")
    rewrite_query: str | None = Field(
        default=None,
        description="A better web search query if relevant is false, otherwise null",
    )


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
