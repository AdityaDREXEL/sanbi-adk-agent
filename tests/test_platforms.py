"""
tests/test_platforms.py — deterministic citation classifier + platform weights.

Pure functions. Zero network, zero LLM.
"""

import pytest

from sanbi_core.platforms import (
    PLATFORM_WEIGHTS,
    classify_domain,
    classify_url,
    platform_weight,
)


# ==============================================================================
# classify_url — exact-domain map
# ==============================================================================
@pytest.mark.parametrize(
    "url,domain,platform",
    [
        ("https://www.reddit.com/r/lasik/comments/abc/x", "reddit.com", "reddit"),
        ("https://old.reddit.com/r/lasik/comments/abc/x", "old.reddit.com", "reddit"),
        ("https://x.com/someuser/status/123", "x.com", "x"),
        ("https://twitter.com/someuser/status/123", "twitter.com", "x"),
        ("https://www.youtube.com/watch?v=abc123", "youtube.com", "youtube"),
        ("https://youtu.be/abc123", "youtu.be", "youtube"),
        ("https://news.ycombinator.com/item?id=1", "news.ycombinator.com", "hn"),
        ("https://www.quora.com/What-is-LASIK", "quora.com", "quora"),
        ("https://stackoverflow.com/questions/1", "stackoverflow.com", "stackexchange"),
        ("https://medium.com/@author/post", "medium.com", "medium"),
        ("https://github.com/owner/repo/issues/1", "github.com", "github"),
        ("https://www.linkedin.com/posts/someone", "linkedin.com", "linkedin"),
        ("https://www.trustpilot.com/review/sight360.com", "trustpilot.com", "reviews"),
        ("https://www.yelp.com/biz/sight360", "yelp.com", "reviews"),
        ("https://www.healthgrades.com/group-directory/sight360", "healthgrades.com", "reviews"),
        ("https://www.realself.com/review/lasik", "realself.com", "reviews"),
    ],
)
def test_exact_domain_map(url, domain, platform):
    assert classify_url(url) == (domain, platform)


# ==============================================================================
# classify_url — parent-domain walk + subdomain prefixes
# ==============================================================================
def test_parent_walk_subdomain():
    # electronics.stackexchange.com → walks up to stackexchange.com
    d, p = classify_url("https://electronics.stackexchange.com/questions/1")
    assert (d, p) == ("electronics.stackexchange.com", "stackexchange")


@pytest.mark.parametrize(
    "url,platform",
    [
        ("https://forum.digikey.com/t/current-sensor/123", "forum"),
        ("https://forums.macrumors.com/threads/x.123/", "forum"),
        ("https://community.st.com/thread/123", "forum"),
        ("https://discuss.pytorch.org/t/x/1", "forum"),
        ("https://blog.cloudflare.com/some-post/", "blog"),
        ("https://wiki.archlinux.org/title/X", "wiki"),
        ("https://learn.microsoft.com/en-us/azure/", "tutorial"),
    ],
)
def test_subdomain_prefix_signals(url, platform):
    _, p = classify_url(url)
    assert p == platform


# ==============================================================================
# classify_url — path-pattern fallback
# ==============================================================================
@pytest.mark.parametrize(
    "url,platform",
    [
        ("https://www.eevblog.com/forum/projects/x/", "forum"),
        ("https://www.allaboutvision.com/blog/lasik-cost/", "blog"),
        ("https://example.com/questions/123-best-lasik", "qa"),
        ("https://mirror.site.com/r/lasik/comments/abc", "forum"),
        ("https://vendor.com/wiki/Main_Page", "wiki"),
        ("https://site.com/reviews/sight360", "reviews"),
        ("https://site.com/article/lasik-guide", "blog"),
        ("https://site.com/tutorials/lasik-prep", "tutorial"),
    ],
)
def test_path_pattern_fallback(url, platform):
    _, p = classify_url(url)
    assert p == platform


# ==============================================================================
# classify_url — blocked + unclassifiable + invalid
# ==============================================================================
@pytest.mark.parametrize(
    "url",
    [
        "https://t.co/abc123",                     # shortener
        "https://bit.ly/xyz",                      # shortener
        "https://docs.python.org/3/",              # blocked subdomain prefix
        "https://support.apple.com/en-us/HT123",   # blocked subdomain prefix
        "https://store.google.com/product/pixel",  # blocked subdomain prefix
    ],
)
def test_blocked_urls_return_no_platform(url):
    _, p = classify_url(url)
    assert p is None


def test_unclassifiable_returns_domain_but_no_platform():
    d, p = classify_url("https://www.allaboutvision.com/lasik/cost/")
    assert d == "allaboutvision.com"
    assert p is None


@pytest.mark.parametrize("url", ["", None])
def test_invalid_inputs(url):
    assert classify_url(url) == (None, None)


def test_url_without_scheme_still_classifies():
    d, p = classify_url("reddit.com/r/lasik/comments/abc/x")
    assert (d, p) == ("reddit.com", "reddit")


def test_blocked_subdomain_beats_path_pattern():
    # docs. prefix is blocked even though path contains /tutorial/
    _, p = classify_url("https://docs.example.com/tutorial/getting-started")
    assert p is None


def test_classify_domain_host_only():
    assert classify_domain("www.reddit.com") == "reddit"
    assert classify_domain("forum.vendor.com") == "forum"
    assert classify_domain("unknown-site.com") is None


# ==============================================================================
# platform_weight — replyability ordering invariants
# ==============================================================================
def test_weight_known_and_fallback():
    assert platform_weight("reddit") == 1.30
    assert platform_weight("wiki") == 0.35
    assert platform_weight("never-heard-of-it") == 0.70


def test_replyability_ordering():
    """Tier invariants the scoring story depends on."""
    assert platform_weight("reddit") > platform_weight("youtube") > platform_weight("blog") > platform_weight("wiki")
    assert platform_weight("forum") > platform_weight("medium")
    assert platform_weight("reviews") > platform_weight("x")


def test_all_weights_positive():
    assert all(w > 0 for w in PLATFORM_WEIGHTS.values())
