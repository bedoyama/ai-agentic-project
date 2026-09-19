"""Pydantic models for structured Gemini output."""

from __future__ import annotations

from typing import Literal

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


class Finding(BaseModel):
    """One grounded claim in a briefing."""

    claim: str = Field(description="A single factual claim")
    source_urls: list[str] = Field(
        description="URLs from the provided source list that support the claim"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence that the claim is supported by those sources",
    )


class SourceRef(BaseModel):
    url: str
    title: str = ""


class Briefing(BaseModel):
    """Cited research briefing."""

    headline: str = Field(description="Short headline")
    summary: str = Field(description="One-paragraph overview grounded in the sources")
    findings: list[Finding] = Field(description="Concrete claims, each with source URLs")
    open_questions: list[str] = Field(
        default_factory=list,
        description="What remains unknown or weakly sourced",
    )
    sources: list[SourceRef] = Field(
        default_factory=list,
        description="Sources used; copy from the provided list, do not invent URLs",
    )


class CriticReport(BaseModel):
    """Whether the briefing's claims are grounded in the collected sources."""

    grounded: bool = Field(description="True if every finding is supported by listed sources")
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims that are unsourced, invented, or cite unknown URLs",
    )
    reason: str = Field(description="Short explanation")
    rewrite_query: str | None = Field(
        default=None,
        description="Better search query if grounded is false, otherwise null",
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
