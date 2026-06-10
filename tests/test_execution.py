"""
tests/test_execution.py — engine routing, citation extraction regexes, parallel isolation.

Every LLM call is mocked. Zero API spend.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import sanbi_core.execution as execution
from sanbi_core.execution import ENGINES, fetch_engine_response, query_all_engines


def _openai_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _fake_openai(content):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(return_value=_openai_response(content)))
        )
    )


# ==============================================================================
# Routing guards
# ==============================================================================
async def test_unknown_engine():
    out = await fetch_engine_response("perplexity", "best lasik")
    assert out["text"] == "Unknown engine: perplexity"
    assert out["citations"] == []


async def test_openai_engine_without_client(monkeypatch):
    monkeypatch.setattr(execution, "openai_client", None)
    out = await fetch_engine_response("openai", "best lasik")
    assert out["text"] == "OpenAI API key missing."
    assert out["citations"] == []


# ==============================================================================
# OpenAI branch — citation extraction from text
# ==============================================================================
async def test_openai_citation_extraction_all_formats(monkeypatch):
    content = (
        "Try [Acme Vision](https://acme.com/lasik) for LASIK. "
        "Sight360 is also good (https://sight360.com). "
        "More info at https://bar.com/guide. "
        "And again https://acme.com/lasik for comparison."   # dup — must dedup
    )
    monkeypatch.setattr(execution, "openai_client", _fake_openai(content))

    out = await fetch_engine_response("openai", "best lasik", "United States")

    urls = [c["url"] for c in out["citations"]]
    assert urls == ["https://acme.com/lasik", "https://sight360.com", "https://bar.com/guide"]
    titles = {c["url"]: c["title"] for c in out["citations"]}
    assert titles["https://acme.com/lasik"] == "Acme Vision"          # markdown title
    assert titles["https://sight360.com"] == "Mentioned in Text"      # paren URL
    assert out["text"] == content


async def test_openai_strips_trailing_punctuation(monkeypatch):
    monkeypatch.setattr(execution, "openai_client", _fake_openai("See https://x.com/page.,)"))
    out = await fetch_engine_response("openai", "q")
    assert out["citations"][0]["url"] == "https://x.com/page"


async def test_openai_none_content(monkeypatch):
    monkeypatch.setattr(execution, "openai_client", _fake_openai(None))
    out = await fetch_engine_response("openai", "q")
    assert out["text"] == ""
    assert out["citations"] == []


async def test_openai_api_exception(monkeypatch):
    broken = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=RuntimeError("rate limit")))
        )
    )
    monkeypatch.setattr(execution, "openai_client", broken)
    out = await fetch_engine_response("openai", "q")
    assert out["text"].startswith("[OpenAI Error]")
    assert out["citations"] == []


async def test_openai_receives_location_in_system_prompt(monkeypatch):
    fake = _fake_openai("answer")
    monkeypatch.setattr(execution, "openai_client", fake)
    await fetch_engine_response("openai", "best lasik", "Philadelphia, PA")
    kwargs = fake.chat.completions.create.await_args.kwargs
    system_msg = kwargs["messages"][0]["content"]
    assert "Philadelphia, PA" in system_msg


# ==============================================================================
# Gemini branch — grounding sources + text-URL fallback
# ==============================================================================
async def test_gemini_citations_merge_grounding_and_text(monkeypatch):
    text = (
        "Sight360 leads the market.\n"
        "### SOURCES\n"
        "- [Foo Review] - https://foo.com/review\n"
        "- https://bar.com\n"
        "- [Internal] - https://vertexaisearch.cloud.google.com/redirect/zzz\n"
    )
    sources = [{"url": "https://real-grounded.com", "title": "Grounded"}]
    monkeypatch.setattr(execution, "gemini_grounded_text", AsyncMock(return_value=(text, sources)))

    out = await fetch_engine_response("gemini", "best lasik", "United States")

    urls = [c["url"] for c in out["citations"]]
    assert urls[0] == "https://real-grounded.com"          # grounding metadata first
    assert "https://foo.com/review" in urls                # text fallback w/ title
    assert "https://bar.com" in urls                       # bare text fallback
    assert all("vertexaisearch" not in u for u in urls)    # internal redirect filtered
    titles = {c["url"]: c["title"] for c in out["citations"]}
    assert titles["https://foo.com/review"] == "Foo Review"


async def test_gemini_dedups_grounding_vs_text(monkeypatch):
    text = "### SOURCES\n- [Dup] - https://same.com/page"
    sources = [{"url": "https://same.com/page", "title": "Original"}]
    monkeypatch.setattr(execution, "gemini_grounded_text", AsyncMock(return_value=(text, sources)))
    out = await fetch_engine_response("gemini", "q")
    assert len(out["citations"]) == 1
    assert out["citations"][0]["title"] == "Original"


async def test_gemini_exception_returns_error_dict(monkeypatch):
    monkeypatch.setattr(execution, "gemini_grounded_text", AsyncMock(side_effect=RuntimeError("vertex down")))
    out = await fetch_engine_response("gemini", "q")
    assert out["text"].startswith("Error:")
    assert out["citations"] == []


async def test_gemini_receives_localized_prompt(monkeypatch):
    mock = AsyncMock(return_value=("answer", []))
    monkeypatch.setattr(execution, "gemini_grounded_text", mock)
    await fetch_engine_response("gemini", "best lasik", "Germany")
    sent = mock.await_args.args[0]
    assert "located in Germany" in sent
    assert "best lasik" in sent


# ==============================================================================
# query_all_engines — parallel fan-out + failure isolation
# ==============================================================================
async def test_query_all_engines_happy(monkeypatch):
    async def fake_fetch(engine, prompt, location="United States"):
        return {"text": f"{engine} says hi", "citations": []}

    monkeypatch.setattr(execution, "fetch_engine_response", fake_fetch)
    out = await query_all_engines("best lasik")
    assert set(out.keys()) == set(ENGINES)
    assert out["openai"]["text"] == "openai says hi"
    assert out["gemini"]["text"] == "gemini says hi"


async def test_query_all_engines_isolates_engine_crash(monkeypatch):
    """One engine raising must not lose the other engine's answer."""
    async def fake_fetch(engine, prompt, location="United States"):
        if engine == "openai":
            raise RuntimeError("openai exploded")
        return {"text": "gemini fine", "citations": []}

    monkeypatch.setattr(execution, "fetch_engine_response", fake_fetch)
    out = await query_all_engines("best lasik")
    assert out["openai"]["text"] == "Error: openai exploded"
    assert out["openai"]["citations"] == []
    assert out["gemini"]["text"] == "gemini fine"


def test_engine_roster():
    assert ENGINES == ["openai", "gemini"]
