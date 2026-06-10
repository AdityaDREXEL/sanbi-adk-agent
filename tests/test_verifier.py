"""
tests/test_verifier.py — URL verification verdicts (anti-hallucination layer).

All HTTP mocked with a fake httpx client. Zero network.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import sanbi_core.verifier as verifier
from sanbi_core.verifier import _path_compatible, verify_url, verify_urls


# ==============================================================================
# _path_compatible — soft-404 redirect detection
# ==============================================================================
@pytest.mark.parametrize(
    "requested,final,compatible",
    [
        ("https://a.com/blog/post-1", "https://a.com/blog/post-1", True),
        ("https://a.com/blog/post-1", "https://a.com/blog/post-1?utm=x", True),
        ("https://a.com/blog/post-1", "https://a.com/", False),          # soft-404 to root
        ("https://a.com/blog/post-1", "https://a.com/pricing/page", False),
        ("https://a.com/", "https://a.com/anywhere/else", True),         # cited root: anything ok
        ("https://a.com/r/lasik/comments/x", "https://b.com/r/lasik/comments/x", True),  # path-only check
    ],
)
def test_path_compatible(requested, final, compatible):
    assert _path_compatible(requested, final) is compatible


# ==============================================================================
# Fake httpx client
# ==============================================================================
class _Resp:
    def __init__(self, status, url="https://x.com", body=None, headers=None):
        self.status_code = status
        self.url = url
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body


class _Client:
    """Programmable fake: maps method calls to canned responses."""

    def __init__(self, head=None, get=None, post=None, raises=None):
        self._head, self._get, self._post, self._raises = head, get, post, raises
        self.head_calls, self.get_calls, self.post_calls = [], [], []

    async def head(self, url, **kw):
        self.head_calls.append(url)
        if self._raises:
            raise self._raises
        return self._head

    async def get(self, url, **kw):
        self.get_calls.append(url)
        if self._raises:
            raise self._raises
        return self._get

    async def post(self, url, **kw):
        self.post_calls.append(url)
        return self._post


# ==============================================================================
# Generic tier
# ==============================================================================
async def test_generic_200_verified():
    c = _Client(head=_Resp(200, "https://site.com/blog/post"))
    out = await verify_url(c, "https://site.com/blog/post")
    assert out["verdict"] == "verified"


async def test_generic_404_hallucinated():
    c = _Client(head=_Resp(404, "https://site.com/missing"))
    out = await verify_url(c, "https://site.com/missing")
    assert out["verdict"] == "hallucinated"


async def test_generic_redirect_to_root_suspicious():
    c = _Client(head=_Resp(200, "https://site.com/"))
    out = await verify_url(c, "https://site.com/blog/deep-post")
    assert out["verdict"] == "suspicious"


async def test_generic_403_falls_back_to_get():
    """Bot-blocked HEAD must retry with GET before giving up."""
    c = _Client(head=_Resp(403, "https://site.com/p"), get=_Resp(200, "https://site.com/p"))
    out = await verify_url(c, "https://site.com/p")
    assert out["verdict"] == "verified"
    assert len(c.get_calls) == 1


async def test_generic_403_on_both_unverifiable():
    c = _Client(head=_Resp(403, "https://site.com/p"), get=_Resp(403, "https://site.com/p"))
    out = await verify_url(c, "https://site.com/p")
    assert out["verdict"] == "unverifiable"


async def test_generic_500_falls_back_to_get_then_unverifiable():
    c = _Client(head=_Resp(503, "https://site.com/p"), get=_Resp(503, "https://site.com/p"))
    out = await verify_url(c, "https://site.com/p")
    assert out["verdict"] == "unverifiable"


async def test_generic_timeout_unverifiable():
    import httpx
    c = _Client(raises=httpx.ConnectError("boom"))
    out = await verify_url(c, "https://site.com/p")
    assert out["verdict"] == "unverifiable"
    assert out["status"] == "timeout"


# ==============================================================================
# YouTube tier — oEmbed
# ==============================================================================
async def test_youtube_200_verified():
    c = _Client(get=_Resp(200))
    out = await verify_url(c, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert out["verdict"] == "verified"
    assert "oembed" in c.get_calls[0]


async def test_youtube_400_means_hallucinated():
    """The production learning: oEmbed returns 400 (not 404) for fake video IDs."""
    c = _Client(get=_Resp(400))
    out = await verify_url(c, "https://www.youtube.com/watch?v=FAKEFAKEFAKE")
    assert out["verdict"] == "hallucinated"


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
    ],
)
async def test_youtube_id_extraction_variants(url):
    c = _Client(get=_Resp(200))
    out = await verify_url(c, url)
    assert out["verdict"] == "verified"


async def test_youtube_channel_url_falls_to_generic():
    """No video id → generic HEAD verification, not oEmbed."""
    c = _Client(head=_Resp(200, "https://www.youtube.com/@channel"))
    out = await verify_url(c, "https://www.youtube.com/@channel")
    assert out["verdict"] == "verified"
    assert c.head_calls  # generic path used


# ==============================================================================
# Reddit tier — OAuth-gated
# ==============================================================================
async def test_reddit_without_creds_unverifiable(monkeypatch):
    """No OAuth creds → honest 'unverifiable', never the HEAD lie."""
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(verifier, "_reddit_token", None)
    c = _Client(head=_Resp(200))                     # HEAD would lie with a 200 shell
    out = await verify_url(c, "https://www.reddit.com/r/lasik/comments/abc123/x")
    assert out["verdict"] == "unverifiable"
    assert not c.head_calls                          # generic tier never touched


async def test_reddit_with_creds_verified(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setattr(verifier, "_reddit_token", None)
    body = [{"data": {"children": [{"data": {"subreddit": "lasik"}}]}}]
    c = _Client(post=_Resp(200, body={"access_token": "tok"}), get=_Resp(200, body=body))
    out = await verify_url(c, "https://www.reddit.com/r/lasik/comments/abc123/x")
    assert out["verdict"] == "verified"


async def test_reddit_404_hallucinated(monkeypatch):
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setattr(verifier, "_reddit_token", "cached-token")
    c = _Client(get=_Resp(404))
    out = await verify_url(c, "https://www.reddit.com/r/lasik/comments/zzz999/x")
    assert out["verdict"] == "hallucinated"


async def test_reddit_subreddit_mismatch_suspicious(monkeypatch):
    """Post exists but in a different subreddit than the AI claimed."""
    monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
    monkeypatch.setenv("REDDIT_CLIENT_SECRET", "secret")
    monkeypatch.setattr(verifier, "_reddit_token", "cached-token")
    body = [{"data": {"children": [{"data": {"subreddit": "ophthalmology"}}]}}]
    c = _Client(get=_Resp(200, body=body))
    out = await verify_url(c, "https://www.reddit.com/r/lasik/comments/abc123/x")
    assert out["verdict"] == "suspicious"


# ==============================================================================
# verify_urls — batch API
# ==============================================================================
async def test_verify_urls_batch_dedup(monkeypatch):
    seen = []

    async def fake_verify(client, url):
        seen.append(url)
        return {"status": 200, "final_url": url, "verdict": "verified"}

    monkeypatch.setattr(verifier, "verify_url", fake_verify)
    out = await verify_urls(["https://a.com/x", "https://a.com/x", "https://b.com/y"])
    assert len(out) == 2
    assert len(seen) == 2                            # duplicate URL verified once
    assert out["https://a.com/x"]["verdict"] == "verified"
