"""
tests/test_growth.py — opportunity materializer scoring + playbook branching.

LLM drafting calls mocked. Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import sanbi_core.growth as growth
from sanbi_core.growth import (
    build_opportunities,
    draft_growth_action,
    get_playbook,
)

REDDIT = "https://reddit.com/r/lasik/comments/x1/thread"
BLOG = "https://blog.example.com/lasik-guide"
YELP = "https://yelp.com/biz/sight360"


def _log_entry(prompt, engine, urls, titles=None):
    return {
        "prompt": prompt,
        "engine": engine,
        "grade": {"cited_sources": urls, "source_titles": titles or {}},
    }


# ==============================================================================
# build_opportunities — scoring formula
# ==============================================================================
def test_production_formula_exact():
    """(engines×25 + prompts×5 + recency×20) × platform_weight, to the decimal."""
    log = [
        _log_entry("best lasik philly", "gemini", [REDDIT, BLOG]),
        _log_entry("best lasik philly", "openai", [REDDIT]),
        _log_entry("lasik cost", "gemini", [REDDIT, YELP]),
    ]
    opps = build_opportunities(log)

    by_url = {o["source_url"]: o for o in opps}
    # reddit: 2 engines, 2 prompts → (50 + 10 + 20) × 1.30 = 104.0
    assert by_url[REDDIT]["score"] == 104.0
    assert by_url[REDDIT]["citation_count"] == 3
    assert by_url[REDDIT]["engines"] == ["gemini", "openai"]
    assert by_url[REDDIT]["prompt_breadth"] == 2
    # yelp: 1 engine, 1 prompt → (25 + 5 + 20) × 1.15 = 57.5
    assert by_url[YELP]["score"] == 57.5
    # blog: 1 engine, 1 prompt → (25 + 5 + 20) × 0.50 = 25.0
    assert by_url[BLOG]["score"] == 25.0
    # sorted descending
    assert [o["score"] for o in opps] == [104.0, 57.5, 25.0]


def test_perplexity_only_counts_half_engine():
    log = [_log_entry("p1", "perplexity", [REDDIT])]
    opps = build_opportunities(log)
    # (0.5×25 + 5 + 20) × 1.30 = 37.5 × 1.30 = 48.75
    assert opps[0]["score"] == 48.75


def test_perplexity_with_other_engine_counts_full():
    log = [
        _log_entry("p1", "perplexity", [REDDIT]),
        _log_entry("p1", "gemini", [REDDIT]),
    ]
    opps = build_opportunities(log)
    # 2 full engines: (50 + 5 + 20) × 1.30 = 97.5
    assert opps[0]["score"] == 97.5


# ==============================================================================
# build_opportunities — input hygiene
# ==============================================================================
def test_empty_log():
    assert build_opportunities([]) == []


def test_skips_unclassifiable_and_blocked_urls():
    log = [_log_entry("p1", "gemini", [
        "https://sight360.com/services",      # own site — not a growth surface
        "https://t.co/abc",                   # blocked shortener
        "",                                   # empty
        None,                                 # null
        REDDIT,                               # the only real opportunity
    ])]
    opps = build_opportunities(log)
    assert len(opps) == 1
    assert opps[0]["source_url"] == REDDIT


def test_handles_missing_grade_fields():
    log = [
        {"prompt": "p", "engine": "gemini", "grade": {}},                       # no cited_sources
        {"prompt": "p", "engine": "gemini", "grade": {"cited_sources": None}},  # null
        {"prompt": None, "engine": None, "grade": {"cited_sources": [REDDIT]}}, # null prompt/engine
    ]
    opps = build_opportunities(log)
    assert len(opps) == 1
    # null engine/prompt → breadth 0: (0 + 0 + 20) × 1.30 = 26.0
    assert opps[0]["score"] == 26.0


def test_title_prefers_real_over_placeholder():
    log = [
        _log_entry("p1", "gemini", [REDDIT], {REDDIT: "Source"}),               # placeholder skipped
        _log_entry("p2", "openai", [REDDIT], {REDDIT: "My LASIK experience"}),  # real kept
    ]
    opps = build_opportunities(log)
    assert opps[0]["title"] == "My LASIK experience"


# ==============================================================================
# get_playbook — platform → motion branching
# ==============================================================================
@pytest.mark.parametrize(
    "platform,action_type",
    [
        ("reddit", "community_reply"),
        ("hn", "community_reply"),
        ("forum", "community_reply"),
        ("qa", "expert_answer"),
        ("quora", "expert_answer"),
        ("stackexchange", "expert_answer"),
        ("youtube", "video_engagement"),
        ("reviews", "review_acquisition"),
        ("blog", "counter_content"),
        ("medium", "counter_content"),
        ("tutorial", "counter_content"),
        ("x", "social_engagement"),
        ("linkedin", "social_engagement"),
        ("github", "social_engagement"),
        ("wiki", "reference_note"),
    ],
)
def test_playbook_branching(platform, action_type):
    assert get_playbook(platform)["action_type"] == action_type


def test_unknown_platform_defaults_to_community_reply():
    assert get_playbook("zorp")["action_type"] == "community_reply"


# ==============================================================================
# draft_growth_action — LLM call mocked
# ==============================================================================
OPP = {
    "source_url": REDDIT,
    "source_domain": "reddit.com",
    "platform": "reddit",
    "title": "My LASIK experience",
    "engines": ["gemini", "openai"],
    "prompts": ["best lasik philly", "lasik cost"],
    "citation_count": 3,
    "prompt_breadth": 2,
    "score": 104.0,
}


async def test_draft_happy_path(monkeypatch):
    drafted = {"action_type": "community_reply", "headline": "h", "draft": "d",
               "why_this_source": "w", "effort": "low"}
    mock = AsyncMock(return_value=drafted)
    monkeypatch.setattr(growth, "robust_json_call", mock)

    out = await draft_growth_action(OPP, "Sight360", "sight360.com", "LASIK")

    assert out == drafted
    prompt_sent = mock.await_args.args[0]
    assert REDDIT in prompt_sent
    assert "Sight360" in prompt_sent
    assert "community_reply" in prompt_sent          # playbook routed by platform
    assert mock.await_args.kwargs.get("use_search") is False  # drafting never searches


async def test_draft_fills_missing_action_type(monkeypatch):
    monkeypatch.setattr(growth, "robust_json_call", AsyncMock(return_value={"headline": "h"}))
    out = await draft_growth_action(OPP, "Sight360", "sight360.com", "LASIK")
    assert out["action_type"] == "community_reply"   # backfilled from playbook


async def test_draft_llm_failure_returns_stub(monkeypatch):
    monkeypatch.setattr(growth, "robust_json_call", AsyncMock(side_effect=RuntimeError("down")))
    out = await draft_growth_action(OPP, "Sight360", "sight360.com", "LASIK")
    assert out["action_type"] == "community_reply"
    assert out["headline"] == "Draft generation failed"
    assert out["draft"] == ""
