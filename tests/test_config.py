from __future__ import annotations

import pytest
from langchain_google_genai import ChatGoogleGenerativeAI

from briefing.config import ConfigError, get_api_key, get_llm, get_model_name


def test_get_api_key_prefers_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    assert get_api_key() == "gemini-secret"


def test_get_api_key_falls_back_to_google(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
    assert get_api_key() == "google-secret"


def test_get_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="Missing Gemini API key"):
        get_api_key()


def test_get_model_name_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert get_model_name() == "gemini-3.8-flash"


def test_get_model_name_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.7-flash")
    assert get_model_name() == "gemini-3.7-flash"


def test_get_llm_builds_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    llm = get_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-3.8-flash"
