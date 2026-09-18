"""Load environment and construct the Gemini chat model."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"


class ConfigError(RuntimeError):
    """Raised when required Gemini configuration is missing."""


def get_api_key() -> str:
    """Return GEMINI_API_KEY, falling back to GOOGLE_API_KEY."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ConfigError(
            "Missing Gemini API key. Set GEMINI_API_KEY (preferred) or GOOGLE_API_KEY."
        )
    return key


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def get_llm() -> ChatGoogleGenerativeAI:
    """Build a ChatGoogleGenerativeAI client. Does not call the network."""
    return ChatGoogleGenerativeAI(
        model=get_model_name(),
        api_key=get_api_key(),
        temperature=1.0,
        max_retries=2,
    )
