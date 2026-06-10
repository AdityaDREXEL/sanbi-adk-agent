"""
tests/test_edge_cases.py — cross-cutting edge cases found in the hardening audit.

Covers:
  - real ADK State integration (delta tracking + JSON-serializability — what a
    persistent SessionService requires)
  - input-validation guards (garbage domains, missing tool_context)
  - wholesale verification failure degrading instead of crashing
  - classifier URL hygiene (ports, userinfo, case)
  - misc boundary conditions (top_n clamps, empty batches, all-hallucinated)

Zero API spend.
"""

import json
from unittest.mock import AsyncMock

import pytest

from google.adk.sessions.state import State

import agents.sanbi_audit.agent as agent_mod
from agents.sanbi_audit.agent import (
    draft_growth_actions,
    find_growth_opportunities,
    generate_audit_prompts,
    query_engines,
)
import mcp_server.server as srv
from sanbi_core.platforms import classify_url
from sanbi_core.verifier import verify_urls

IDENTITY = {
    "brand_name": "Sight360",
    "industry": "LASIK & Vision Correction",
    "target_audience": "Adults",
    "competitors": ["LasikPlus"],
}
PROMPTS = [
    {"search_query": "Sight360 reviews 2026", "topic": "LASIK", "prompt_type": "branded", "location": "United States"},
]
REDDIT_URL = "https://reddit.com/r/lasik/comments/x1/thread"


class FakeToolContext:
    def __init__(self, state=None):
        self.state = state if state is not None else {}


def _mock_planning(monkeypatch):
    monkeypatch.setattr(agent_mod, "analyze_brand_identity", AsyncMock(return_value=dict(IDENTITY)))
    monkeypatch.setattr(agent_mod, "auto_generate_brand_prompts", AsyncMock(return_value=[dict(p) for p in PROMPTS]))


# ==============================================================================
# Real ADK State — the persistence contract
# ==============================================================================
async def test_tools_work_with_real_adk_state(monkeypatch):
    """Run tools 1-2 against ADK's actual State class, not a plain dict."""
    _mock_planning(monkeypatch)
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "x" * 30, "citations": []},
        "gemini": {"text": "y" * 30, "citations": []},
    }))
    value, delta = {}, {}
    ctx = FakeToolContext(state=State(value=value, delta=delta))

    out1 = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    audit_id = out1["audit_id"]

    # Assignment must be delta-tracked — this is what a persistent
    # SessionService commits. In-place mutation would leave delta empty.
    assert f"audit:{audit_id}" in delta
    assert "active_audit_id" in delta

    out2 = await query_engines(tool_context=ctx)            # fallback via real State.get
    assert out2["responses_collected"] == 2

    # Everything written to state must survive JSON round-tripping
    # (DatabaseSessionService serializes state as JSON).
    serialized = json.dumps(delta)
    assert len(json.loads(serialized)[f"audit:{audit_id}"]["responses"]) == 2


async def test_state_updates_visible_after_save(monkeypatch):
    """Read-modify-write through the real State: second read sees tool-2 data."""
    _mock_planning(monkeypatch)
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "x" * 30, "citations": []},
        "gemini": {"text": "y" * 30, "citations": []},
    }))
    ctx = FakeToolContext(state=State(value={}, delta={}))
    out1 = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    await query_engines(tool_context=ctx)
    stored = ctx.state[f"audit:{out1['audit_id']}"]
    assert len(stored["responses"]) == 2


# ==============================================================================
# Input-validation guards
# ==============================================================================
@pytest.mark.parametrize("bad_domain", ["", "   ", "sight360", "https://", "not a domain", None])
async def test_generate_audit_prompts_rejects_garbage_domains(monkeypatch, bad_domain):
    research = AsyncMock()
    monkeypatch.setattr(agent_mod, "analyze_brand_identity", research)
    out = await generate_audit_prompts(bad_domain, "LASIK", tool_context=FakeToolContext())
    assert "error" in out
    research.assert_not_awaited()              # the money guard: no LLM on garbage


async def test_generate_audit_prompts_accepts_url_style_domain(monkeypatch):
    _mock_planning(monkeypatch)
    out = await generate_audit_prompts("https://sight360.com/", "LASIK", tool_context=FakeToolContext())
    assert "error" not in out
    kwargs = agent_mod.analyze_brand_identity.await_args.args
    assert kwargs[0] == "https://sight360.com"  # trailing slash stripped


async def test_tools_survive_missing_tool_context(monkeypatch):
    """If injection ever fails, tools must return an error dict, not raise."""
    _mock_planning(monkeypatch)
    out1 = await generate_audit_prompts("sight360.com", "LASIK", tool_context=None)
    assert "error" in out1
    out2 = await query_engines("someid", tool_context=None)
    assert "error" in out2
    out3 = await find_growth_opportunities(tool_context=None)
    assert "error" in out3
    out4 = await draft_growth_actions(tool_context=None)
    assert "error" in out4


@pytest.mark.parametrize("bad_domain", ["", "nodot", "has space.com x"])
async def test_mcp_rejects_garbage_domains(monkeypatch, bad_domain):
    research = AsyncMock()
    monkeypatch.setattr(srv, "analyze_brand_identity", research)
    out = await srv.run_visibility_audit(bad_domain, "LASIK")
    assert "error" in out
    research.assert_not_awaited()


# ==============================================================================
# Wholesale verification failure — degrade, don't crash
# ==============================================================================
def _seed_graded(ctx):
    aid = "edgeaudit0001"
    ctx.state[f"audit:{aid}"] = {
        "domain": "sight360.com", "brand_name": "Sight360", "topic": "LASIK",
        "location": "United States", "identity": dict(IDENTITY),
        "prompts": [dict(p) for p in PROMPTS],
        "responses": [{"prompt": "p", "prompt_type": "b", "engine": "gemini", "response": {}}],
        "audit_log": [
            {"prompt": "p1", "prompt_type": "unbranded", "engine": "gemini",
             "grade": {"is_visible": False, "cited_sources": [REDDIT_URL], "source_titles": {}}},
        ],
    }
    ctx.state["active_audit_id"] = aid
    return aid


async def test_find_growth_survives_verifier_meltdown(monkeypatch):
    ctx = FakeToolContext()
    aid = _seed_graded(ctx)
    monkeypatch.setattr(agent_mod, "verify_urls", AsyncMock(side_effect=RuntimeError("network down")))

    out = await find_growth_opportunities(aid, tool_context=ctx)

    assert "error" not in out
    assert out["total_opportunities"] == 1
    assert out["top_opportunities"][0]["url_verdict"] == "unverifiable"


async def test_mcp_survives_verifier_meltdown(monkeypatch):
    monkeypatch.setattr(srv, "analyze_brand_identity", AsyncMock(return_value=dict(IDENTITY)))
    monkeypatch.setattr(srv, "auto_generate_brand_prompts", AsyncMock(return_value=[dict(p) for p in PROMPTS]))
    monkeypatch.setattr(srv, "query_all_engines", AsyncMock(return_value={
        "gemini": {"text": "t" * 30, "citations": []},
    }))
    monkeypatch.setattr(srv, "grade_result", AsyncMock(return_value={
        "is_visible": True, "rank": 1, "sentiment": "Positive", "visibility_score": 100,
        "sentiment_score": 100, "cited_sources": [REDDIT_URL], "source_titles": {},
        "ranking_table": [],
    }))
    monkeypatch.setattr(srv, "generate_executive_summary", AsyncMock(
        return_value={"positioning": "ok", "key_selling_points": [], "negative_risks": []}))
    monkeypatch.setattr(srv, "verify_urls", AsyncMock(side_effect=RuntimeError("network down")))

    out = await srv.run_visibility_audit("sight360.com", "LASIK")
    assert "error" not in out
    assert out["growth_inbox"]["top_opportunities"][0]["url_verdict"] == "unverifiable"


# ==============================================================================
# Classifier URL hygiene (ports / userinfo / case — all seen in real citations)
# ==============================================================================
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://reddit.com:443/r/lasik/comments/x/t", ("reddit.com", "reddit")),
        ("https://user@reddit.com/r/lasik/comments/x/t", ("reddit.com", "reddit")),
        ("HTTPS://REDDIT.COM/r/lasik/comments/x/t", ("reddit.com", "reddit")),
        ("https://WWW.Yelp.com:8443/biz/sight360", ("yelp.com", "reviews")),
    ],
)
def test_classifier_strips_port_userinfo_case(url, expected):
    assert classify_url(url) == expected


# ==============================================================================
# Misc boundaries
# ==============================================================================
async def test_verify_urls_empty_batch():
    assert await verify_urls([]) == {}


async def test_draft_top_n_zero_clamps_to_one(monkeypatch):
    ctx = FakeToolContext()
    aid = _seed_graded(ctx)
    audit = ctx.state[f"audit:{aid}"]
    audit["opportunities"] = [
        {"source_url": f"https://reddit.com/r/x/comments/{i}/t", "platform": "reddit",
         "title": None, "engines": ["gemini"], "prompts": ["p"], "citation_count": 1,
         "prompt_breadth": 1, "score": 50.0, "url_verdict": "verified"}
        for i in range(3)
    ]
    monkeypatch.setattr(agent_mod, "draft_growth_action", AsyncMock(return_value={
        "action_type": "community_reply", "headline": "h", "draft": "d",
        "why_this_source": "w", "effort": "low"}))
    out = await draft_growth_actions(aid, top_n=0, tool_context=ctx)
    assert out["drafts_generated"] == 1


async def test_draft_all_hallucinated_yields_zero_drafts(monkeypatch):
    ctx = FakeToolContext()
    aid = _seed_graded(ctx)
    ctx.state[f"audit:{aid}"]["opportunities"] = [
        {"source_url": REDDIT_URL, "platform": "reddit", "title": None,
         "engines": ["gemini"], "prompts": ["p"], "citation_count": 1,
         "prompt_breadth": 1, "score": 50.0, "url_verdict": "hallucinated"},
    ]
    drafter = AsyncMock()
    monkeypatch.setattr(agent_mod, "draft_growth_action", drafter)
    out = await draft_growth_actions(aid, tool_context=ctx)
    assert out["drafts_generated"] == 0
    assert out["drafts"] == []
    drafter.assert_not_awaited()               # zero LLM spend on fakes


async def test_location_empty_string_gets_default(monkeypatch):
    _mock_planning(monkeypatch)
    ctx = FakeToolContext()
    out = await generate_audit_prompts("sight360.com", "LASIK", location="", tool_context=ctx)
    assert ctx.state[f"audit:{out['audit_id']}"]["location"] == "United States"
