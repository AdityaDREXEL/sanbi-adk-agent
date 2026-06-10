"""
sanbi_core/verifier.py — Citation URL verification (anti-hallucination layer).

Ported from production Sanbi `growth/url_verifier.py`. AI engines sometimes
cite URLs that don't exist. Before we tell a marketing team "go reply to this
thread", we prove the thread is real.

Verdicts:
    verified      — URL is live and the path matches what was claimed
    suspicious    — URL responds but redirected somewhere materially different
    hallucinated  — URL does not exist (404/410, or invalid YouTube video id)
    unverifiable  — bot-blocked / timeout / missing creds; can't prove either way

Production learnings baked in (discovered the hard way, June 2026):
  - Reddit blocks ALL unauthenticated API access: .json returns 403 for every
    UA, and a plain HEAD returns a 200 SPA shell EVEN FOR FAKE POST IDS — a
    false positive. Only OAuth works → gated on REDDIT_CLIENT_ID/SECRET env
    vars; without creds we honestly return "unverifiable".
  - YouTube oEmbed returns 400 (not 404) for invalid video IDs.
  - httpx: max_redirects is a client-constructor option, not per-request.
"""

import asyncio
import logging
import os
import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

VERIFY_TIMEOUT = 10.0
_UA = "Mozilla/5.0 (compatible; SanbiAuditBot/1.0; +https://sanbi.ai)"

# Statuses that mean "a bot wall, not a missing page".
_BLOCKED_STATUSES = {401, 403, 405, 429, 999}


def _path_compatible(requested: str, final: str) -> bool:
    """Did redirects keep us in the same section of the site?

    Compares the first two path segments. A deep URL that redirects to the
    site root (or a totally different section) is 'suspicious' — the page the
    AI cited probably doesn't exist, the server just soft-404'd it.
    """
    try:
        rp = [s for s in urlparse(requested).path.split("/") if s][:2]
        fp = [s for s in urlparse(final).path.split("/") if s][:2]
    except Exception:
        return True
    if not rp:
        return True  # cited the root; anywhere is fine
    return rp == fp


# ------------------------------------------------------------------------------
# Tier 1a — Reddit (OAuth only; everything else lies)
# ------------------------------------------------------------------------------
_reddit_token: Optional[str] = None


async def _get_reddit_token(client: httpx.AsyncClient) -> Optional[str]:
    global _reddit_token
    if _reddit_token:
        return _reddit_token
    cid = os.getenv("REDDIT_CLIENT_ID")
    secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not cid or not secret:
        return None
    try:
        r = await client.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=(cid, secret),
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": _UA},
        )
        if r.status_code == 200:
            _reddit_token = r.json().get("access_token")
            return _reddit_token
    except Exception as e:
        logger.warning(f"Reddit token fetch failed: {e}")
    return None


async def _verify_reddit(client: httpx.AsyncClient, url: str) -> dict:
    token = await _get_reddit_token(client)
    if not token:
        # No creds: a generic HEAD would return a 200 SPA shell even for fake
        # posts (false positive), so we refuse to guess.
        return {"status": "blocked", "final_url": url, "verdict": "unverifiable"}

    m = re.search(r"/comments/([a-z0-9]+)", url, re.IGNORECASE)
    if not m:
        return await _verify_generic(client, url)
    post_id = m.group(1)

    try:
        r = await client.get(
            f"https://oauth.reddit.com/comments/{post_id}.json",
            headers={"Authorization": f"Bearer {token}", "User-Agent": _UA},
            params={"limit": 1},
        )
        if r.status_code == 200:
            # Cross-check the claimed subreddit against reality.
            claimed = re.search(r"/r/([^/]+)/", url, re.IGNORECASE)
            try:
                real_sub = r.json()[0]["data"]["children"][0]["data"]["subreddit"]
            except Exception:
                real_sub = None
            if claimed and real_sub and claimed.group(1).lower() != real_sub.lower():
                return {"status": 200, "final_url": url, "verdict": "suspicious"}
            return {"status": 200, "final_url": url, "verdict": "verified"}
        if r.status_code in (404, 410):
            return {"status": r.status_code, "final_url": url, "verdict": "hallucinated"}
        return {"status": r.status_code, "final_url": url, "verdict": "unverifiable"}
    except Exception:
        return {"status": "error", "final_url": url, "verdict": "unverifiable"}


# ------------------------------------------------------------------------------
# Tier 1b — YouTube (oEmbed: free, no API key)
# ------------------------------------------------------------------------------
_YT_ID_PATTERNS = (
    r"[?&]v=([A-Za-z0-9_-]{6,})",
    r"youtu\.be/([A-Za-z0-9_-]{6,})",
    r"/shorts/([A-Za-z0-9_-]{6,})",
    r"/embed/([A-Za-z0-9_-]{6,})",
)


async def _verify_youtube(client: httpx.AsyncClient, url: str) -> dict:
    video_id = None
    for pat in _YT_ID_PATTERNS:
        m = re.search(pat, url)
        if m:
            video_id = m.group(1)
            break
    if not video_id:
        return await _verify_generic(client, url)  # channel/playlist URL

    try:
        r = await client.get(
            "https://www.youtube.com/oembed",
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
        )
        if r.status_code == 200:
            return {"status": 200, "final_url": url, "verdict": "verified"}
        if r.status_code in (400, 404, 410):  # 400 = invalid video id!
            return {"status": r.status_code, "final_url": url, "verdict": "hallucinated"}
        return {"status": r.status_code, "final_url": url, "verdict": "unverifiable"}
    except Exception:
        return {"status": "error", "final_url": url, "verdict": "unverifiable"}


# ------------------------------------------------------------------------------
# Tier 2 — Generic HEAD (everything else)
# ------------------------------------------------------------------------------
async def _verify_generic(client: httpx.AsyncClient, url: str) -> dict:
    try:
        r = await client.head(url)
        if r.status_code in _BLOCKED_STATUSES or r.status_code >= 500:
            r = await client.get(url)  # many servers reject HEAD but allow GET
        status, final = r.status_code, str(r.url)

        if 200 <= status < 300:
            verdict = "verified" if _path_compatible(url, final) else "suspicious"
            return {"status": status, "final_url": final, "verdict": verdict}
        if status in (404, 410):
            return {"status": status, "final_url": final, "verdict": "hallucinated"}
        if status in _BLOCKED_STATUSES:
            return {"status": status, "final_url": final, "verdict": "unverifiable"}
        return {"status": status, "final_url": final, "verdict": "unverifiable"}
    except (httpx.TimeoutException, httpx.ConnectError):
        return {"status": "timeout", "final_url": url, "verdict": "unverifiable"}
    except Exception:
        return {"status": "error", "final_url": url, "verdict": "unverifiable"}


# ------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------
async def verify_url(client: httpx.AsyncClient, url: str) -> dict:
    """Route a URL to the right verification tier."""
    host = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
    if "reddit.com" in host:
        return await _verify_reddit(client, url)
    if "youtube.com" in host or "youtu.be" in host:
        return await _verify_youtube(client, url)
    return await _verify_generic(client, url)


async def verify_urls(urls: List[str], concurrency: int = 10) -> Dict[str, dict]:
    """Verify a batch of URLs concurrently. Returns {url: {status, final_url, verdict}}."""
    results: Dict[str, dict] = {}
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
        follow_redirects=True,
        max_redirects=8,  # constructor option — NOT a per-request kwarg
        timeout=VERIFY_TIMEOUT,
        headers={"User-Agent": _UA},
    ) as client:
        async def _one(u: str):
            async with sem:
                results[u] = await verify_url(client, u)

        await asyncio.gather(*(_one(u) for u in dict.fromkeys(urls)))

    return results
