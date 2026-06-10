"""
sanbi_core/platforms.py — Citation source classification.

Verbatim port of production Sanbi `growth/platforms.py`: the single source of
truth for classifying a cited URL into a platform bucket. Deterministic and
free — no LLM call needed. Battle-tested against ~40k real citations.

Design:
  - Strip "www." prefix correctly (str.lstrip strips a CHARSET, not a prefix).
  - Match against an exact-domain map first.
  - If that fails, walk up the domain (e.g. "electronics.stackexchange.com" →
    "stackexchange.com") so subdomains are normalized.
  - Then subdomain-prefix signals (forum./blog./community.) and URL-path
    patterns (/forum/, /questions/, /blog/) catch the long tail of domains
    we've never seen.
  - Return None for non-community domains so the materializer skips them.
"""

from typing import Optional
from urllib.parse import urlparse


# -----------------------------------------------------------------------------
# Canonical domain → platform-bucket map.
# Keys are bare hostnames (no scheme, no path, no "www.").
# -----------------------------------------------------------------------------
PLATFORM_BY_DOMAIN: dict[str, str] = {
    # Reddit
    "reddit.com":              "reddit",
    "old.reddit.com":          "reddit",
    "new.reddit.com":          "reddit",
    "i.reddit.com":            "reddit",
    "np.reddit.com":           "reddit",

    # X / Twitter
    "x.com":                   "x",
    "twitter.com":             "x",
    "mobile.twitter.com":      "x",
    "nitter.net":              "x",

    # Meta family
    "instagram.com":           "instagram",
    "facebook.com":            "facebook",
    "m.facebook.com":          "facebook",

    # TikTok
    "tiktok.com":              "tiktok",
    "vm.tiktok.com":           "tiktok",

    # YouTube
    "youtube.com":             "youtube",
    "m.youtube.com":           "youtube",
    "youtu.be":                "youtube",

    # Q&A / knowledge
    "quora.com":               "quora",
    "stackoverflow.com":       "stackexchange",
    "stackexchange.com":       "stackexchange",
    "superuser.com":           "stackexchange",
    "serverfault.com":         "stackexchange",
    "mathoverflow.net":        "stackexchange",
    "askubuntu.com":           "stackexchange",

    # Hacker News
    "news.ycombinator.com":    "hn",

    # Blogs / pubs
    "medium.com":              "medium",
    "dev.to":                  "devto",
    "substack.com":            "medium",  # treat as similar bucket

    # Code / product
    "github.com":              "github",
    "gist.github.com":         "github",
    "producthunt.com":         "producthunt",

    # Pro / business
    "linkedin.com":            "linkedin",

    # Reviews (high-intent — often top citation source for B2C)
    "trustpilot.com":          "reviews",
    "nl.trustpilot.com":       "reviews",
    "uk.trustpilot.com":       "reviews",
    "feedbackcompany.com":     "reviews",
    "g2.com":                  "reviews",
    "capterra.com":            "reviews",
    "trustradius.com":         "reviews",
    "sitejabber.com":          "reviews",
    "yelp.com":                "reviews",
    "healthgrades.com":        "reviews",
    "realself.com":            "reviews",
}


# Domains we explicitly don't want as Growth opportunities even though they
# look "social-ish" (URL shorteners, internal redirects, dead-end pages).
_BLOCKED = {
    "t.co",      # twitter shortener — would need to resolve
    "lnkd.in",   # linkedin shortener
    "bit.ly",
    "tinyurl.com",
    "goo.gl",
}

# -----------------------------------------------------------------------------
# Dynamic pattern signals — classify domains we've never seen before, instead
# of maintaining a static allowlist that always lags reality.
#
# Order of evaluation (in classify_url):
#   1.  Blocked exact host                 → None
#   2.  Blocked subdomain prefix           → None   (docs/dev/help/store/jobs/...)
#   3.  PLATFORM_BY_DOMAIN exact match     → brand override (reddit/quora/...)
#   4.  PLATFORM_BY_DOMAIN parent-walk     → brand override
#   5.  Subdomain prefix → platform        → forum/blog/qa/...
#   6.  URL path pattern → platform        → forum/blog/qa/wiki/...
#   7.  None
# -----------------------------------------------------------------------------

# 1st-party content surfaces that aren't engagement opportunities. Auto-drop.
_BLOCKED_SUBDOMAIN_PREFIXES: tuple[str, ...] = (
    "docs.", "developer.", "developers.", "help.", "support.", "kb.",
    "api.", "status.", "careers.", "jobs.", "press.", "investors.",
    "store.", "shop.", "buy.", "cdn.", "static.", "assets.", "media.",
    "images.", "img.",
)

# Subdomain prefix → platform bucket. Catches the long tail of vendor sites:
#   forum.digikey.com → forum     blog.cloudflare.com → blog
#   community.st.com  → forum     news.ycombinator.com is overridden by exact map
_SUBDOMAIN_PREFIX_PLATFORM: dict[str, str] = {
    "forum.":     "forum",
    "forums.":    "forum",
    "community.": "forum",
    "discuss.":   "forum",
    "blog.":      "blog",
    "blogs.":     "blog",
    "news.":      "blog",
    "wiki.":      "wiki",
    "learn.":     "tutorial",
    "academy.":   "tutorial",
}

# URL path substring → platform bucket. Matched against the lowercased path
# only when host classification fell through. First match wins, so order the
# more specific patterns first.
_PATH_PATTERN_PLATFORM: list[tuple[str, str]] = [
    # explicit community/forum paths
    ("/forum/",        "forum"),
    ("/forums/",       "forum"),
    ("/community/",    "forum"),
    ("/discussion/",   "forum"),
    ("/discuss/",      "forum"),
    ("/thread/",       "forum"),
    ("/threads/",      "forum"),
    ("/topic/",        "forum"),
    # Q&A
    ("/questions/",    "qa"),
    ("/question/",     "qa"),
    ("/ask/",          "qa"),
    # reddit-style comment threads on mirror sites
    ("/r/",            "forum"),
    ("/comments/",     "forum"),
    # blogs & editorial (vendor + 3rd-party publications)
    ("/blog/",         "blog"),
    ("/blogs/",        "blog"),
    ("/post/",         "blog"),
    ("/posts/",        "blog"),
    ("/article/",      "blog"),
    ("/articles/",     "blog"),
    ("/news/",         "blog"),
    ("/whitepaper",    "blog"),
    # reference material
    ("/wiki/",         "wiki"),
    ("/learn/",        "tutorial"),
    ("/tutorial/",     "tutorial"),
    ("/tutorials/",    "tutorial"),
    # reviews
    ("/review/",       "reviews"),
    ("/reviews/",      "reviews"),
]


def _strip_www(host: str) -> str:
    """Strip a literal 'www.' prefix (str.lstrip strips a charset, not a prefix)."""
    h = (host or "").lower().strip()
    if h.startswith("www."):
        h = h[4:]
    return h


def _is_blocked_host(host: str) -> bool:
    """Blocked shorteners + first-party doc/store/jobs subdomains.

    Blocked-ness must also veto the URL-path fallback in classify_url —
    docs.example.com/tutorial/x is still a docs page, not a tutorial
    opportunity.
    """
    if not host or host in _BLOCKED:
        return True
    return any(host.startswith(b) for b in _BLOCKED_SUBDOMAIN_PREFIXES)


def _walk_up(host: str):
    """Yield host and successive parent domains: a.b.c → a.b.c, b.c."""
    parts = host.split(".")
    for i in range(len(parts) - 1):
        yield ".".join(parts[i:])


def _classify_host(host: str) -> Optional[str]:
    """Stage 1-5: classify based on hostname alone (no path)."""
    if _is_blocked_host(host):
        return None

    # Brand override: exact match.
    hit = PLATFORM_BY_DOMAIN.get(host)
    if hit:
        return hit

    # Brand override: walk up subdomains.
    for parent in _walk_up(host):
        hit = PLATFORM_BY_DOMAIN.get(parent)
        if hit:
            return hit

    # Subdomain prefix → platform.
    for prefix, platform in _SUBDOMAIN_PREFIX_PLATFORM.items():
        if host.startswith(prefix):
            return platform

    return None


def classify_domain(domain: str) -> Optional[str]:
    """Host-only classifier. Prefer classify_url() which also inspects paths."""
    return _classify_host(_strip_www(domain))


def classify_url(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Full classifier: returns (domain, platform). Uses host signals first; if
    those don't match, falls back to URL-path patterns so we catch blogs and
    forums hosted on vendor root domains (e.g. eevblog.com/forum/...).
    Returns (None, None) for invalid URLs, (domain, None) for unclassifiable.
    """
    if not url:
        return None, None
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
    except Exception:
        return None, None
    # .hostname (vs .netloc) strips the port and userinfo and lowercases —
    # "User@Reddit.COM:443" → "reddit.com". Real citations contain all three.
    host = _strip_www(parsed.hostname or "")
    if not host:
        return None, None

    platform = _classify_host(host)
    if platform:
        return host, platform

    # Path-pattern fallback. Only fires when host signals all whiffed AND the
    # host isn't explicitly blocked (docs./store./shorteners stay dropped).
    if not _is_blocked_host(host):
        path = (parsed.path or "").lower()
        if path:
            for needle, plat in _PATH_PATTERN_PLATFORM:
                if needle in path:
                    return host, plat

    return host, None


# -----------------------------------------------------------------------------
# Platform scoring weights.
#
# Heuristic: weight ≈ "probability a reply here actually converts."
# Two-engine, high-citation entries win regardless of platform, but at equal
# raw score we prefer platforms where (a) we CAN reply at all and (b) the
# audience is high-intent.
#
#   - blog/tutorial/wiki rank LOW because most cited blog/news/wiki pages
#     don't accept third-party comments. They're still surfaced for
#     counter-content opportunities, but shouldn't crowd out replyable
#     surfaces at the top of the inbox.
#   - youtube outranks blog because you CAN comment on a video.
#   - reddit/forum/hn dominate because they're explicit Q&A surfaces.
# -----------------------------------------------------------------------------
PLATFORM_WEIGHTS: dict[str, float] = {
    # Tier 1 — explicit Q&A / discussion surfaces (reply = direct conversion)
    "reddit":         1.30,
    "hn":             1.25,
    "forum":          1.25,
    "stackexchange":  1.20,
    "qa":             1.10,
    "quora":          1.10,
    "reviews":        1.15,   # high commercial intent (Trustpilot, G2, Yelp)

    # Tier 2 — comment-supported social
    "x":              1.00,
    "linkedin":       1.00,
    "youtube":        0.85,
    "instagram":      0.65,
    "tiktok":         0.65,
    "facebook":       0.60,

    # Tier 3 — code / product communities
    "github":         0.75,
    "producthunt":    0.80,
    "devto":          0.75,

    # Tier 4 — editorial / reference (often NOT commentable; counter-content only)
    "medium":         0.70,
    "blog":           0.50,
    "tutorial":       0.45,
    "wiki":           0.35,

    # Fallback
    "other_community": 0.70,
}


def platform_weight(platform: str) -> float:
    return PLATFORM_WEIGHTS.get(platform, 0.70)
