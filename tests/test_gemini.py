"""
tests/test_gemini.py — fence stripping, grounding URL resolution, client guards.

Every network call is mocked. Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import sanbi_core.gemini as gemini
from sanbi_core.gemini import (
    _resolve_grounding_urls,
    _resolve_one,
    _strip_code_fences,
    gemini_force_json,
    gemini_grounded_text,
    robust_json_call,
)


# ==============================================================================
# _strip_code_fences — every fence style the models emit
# ==============================================================================
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),                                  # no fences
        ('```json\n{"a": 1}\n```', '{"a": 1}'),                    # json fence
        ('```JSON\n{"a": 1}\n```', '{"a": 1}'),                    # uppercase tag
        ('```\n{"a": 1}\n```', '{"a": 1}'),                        # bare fence (pre-fix bug)
        ('```python\n{"a": 1}\n```', '{"a": 1}'),                  # wrong language tag
        ('  ```json\n{"a": 1}\n```  ', '{"a": 1}'),                # surrounding whitespace
        ("", ""),
        (None, ""),
    ],
)
def test_strip_code_fences(raw, expected):
    assert _strip_code_fences(raw) == expected


# ==============================================================================
# _resolve_one — vertexaisearch redirect resolution
# ==============================================================================
class _FakeResp:
    def __init__(self, location=None):
        self.headers = {"location": location} if location else {}


class _FakeClient:
    def __init__(self, location=None, raises=False):
        self._location = location
        self._raises = raises

    async def get(self, uri, follow_redirects=False, timeout=None):
        if self._raises:
            raise RuntimeError("network down")
        return _FakeResp(self._location)


async def test_resolve_one_follows_redirect():
    client = _FakeClient(location="https://actual-page.com/article")
    out = await _resolve_one(client, "https://vertexaisearch.cloud.google.com/x", "title")
    assert out == "https://actual-page.com/article"


async def test_resolve_one_falls_back_to_bare_domain_title():
    client = _FakeClient(location=None)
    out = await _resolve_one(client, "https://vertexaisearch.cloud.google.com/x", "ti.com")
    assert out == "https://ti.com"


async def test_resolve_one_keeps_uri_for_non_domain_title():
    client = _FakeClient(location=None)
    uri = "https://vertexaisearch.cloud.google.com/x"
    out = await _resolve_one(client, uri, "TI Official Site")
    assert out == uri


async def test_resolve_one_network_error_uses_title_fallback():
    client = _FakeClient(raises=True)
    out = await _resolve_one(client, "https://vertexaisearch.cloud.google.com/x", "example.org")
    assert out == "https://example.org"


async def test_resolve_grounding_urls_empty_list():
    assert await _resolve_grounding_urls([]) == []


# ==============================================================================
# Client-not-initialized guards (also guarantees tests never hit the network)
# ==============================================================================
async def test_grounded_text_without_client(monkeypatch):
    monkeypatch.setattr(gemini, "gemini_client", None)
    text, sources = await gemini_grounded_text("anything", use_search=True)
    assert text == ""
    assert sources == []


async def test_force_json_without_client(monkeypatch):
    monkeypatch.setattr(gemini, "gemini_client", None)
    assert await gemini_force_json("ctx", "schema") == {}


async def test_single_json_call_without_client(monkeypatch):
    monkeypatch.setattr(gemini, "gemini_client", None)
    assert await robust_json_call("p", "i", use_search=False) == {}


# ==============================================================================
# robust_json_call — routing logic
# ==============================================================================
async def test_robust_json_no_search_routes_to_single_call(monkeypatch):
    single = AsyncMock(return_value={"x": 1})
    grounded = AsyncMock()
    monkeypatch.setattr(gemini, "_single_json_call", single)
    monkeypatch.setattr(gemini, "gemini_grounded_text", grounded)

    out = await robust_json_call("prompt", "instruction", use_search=False)
    assert out == {"x": 1}
    single.assert_awaited_once_with("prompt", "instruction")
    grounded.assert_not_awaited()


async def test_robust_json_with_search_two_step(monkeypatch):
    monkeypatch.setattr(gemini, "gemini_grounded_text", AsyncMock(return_value=("research text", [])))
    force = AsyncMock(return_value={"y": 2})
    monkeypatch.setattr(gemini, "gemini_force_json", force)

    out = await robust_json_call("prompt", "instruction", use_search=True)
    assert out == {"y": 2}
    force.assert_awaited_once_with("research text", "instruction")


async def test_robust_json_with_search_empty_research_short_circuits(monkeypatch):
    monkeypatch.setattr(gemini, "gemini_grounded_text", AsyncMock(return_value=("", [])))
    force = AsyncMock()
    monkeypatch.setattr(gemini, "gemini_force_json", force)

    out = await robust_json_call("prompt", "instruction", use_search=True)
    assert out == {}
    force.assert_not_awaited()                # never pay for formatting an empty context
