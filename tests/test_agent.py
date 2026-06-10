"""
tests/test_agent.py — multi-agent tree + session-state tool flow.

Tools receive a FakeToolContext (plain-dict .state) — the only ToolContext
surface they touch. sanbi_core functions are mocked at the agent-module
level. Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import agents.sanbi_audit.agent as agent_mod
from agents.sanbi_audit.agent import (
    audit_agent,
    draft_growth_actions,
    find_growth_opportunities,
    generate_audit_prompts,
    grade_responses,
    growth_agent,
    query_engines,
    root_agent,
)

IDENTITY = {
    "brand_name": "Sight360",
    "industry": "LASIK & Vision Correction",
    "target_audience": "Adults considering vision correction",
    "competitors": ["LasikPlus", "TLC", "ClearChoice", "EyeMed", "VisionWorks", "Extra6"],
}

PROMPTS = [
    {"search_query": "Sight360 reviews 2026", "topic": "LASIK", "prompt_type": "branded", "location": "United States"},
    {"search_query": "best lasik clinics philadelphia", "topic": "LASIK", "prompt_type": "unbranded", "location": "United States"},
]


class FakeToolContext:
    """Stand-in for ADK ToolContext — tools only use the dict-like .state."""

    def __init__(self):
        self.state = {}


@pytest.fixture
def ctx():
    return FakeToolContext()


def _mock_planning(monkeypatch):
    monkeypatch.setattr(agent_mod, "analyze_brand_identity", AsyncMock(return_value=dict(IDENTITY)))
    monkeypatch.setattr(agent_mod, "auto_generate_brand_prompts", AsyncMock(return_value=[dict(p) for p in PROMPTS]))


# ==============================================================================
# Agent tree contract
# ==============================================================================
def test_agent_tree_definition():
    assert root_agent.name == "sanbi_coordinator"
    assert len(root_agent.tools) == 0                       # coordinator only routes
    assert [a.name for a in root_agent.sub_agents] == ["audit_agent", "growth_agent"]

    assert {t.__name__ for t in audit_agent.tools} == {
        "generate_audit_prompts", "query_engines", "grade_responses",
    }
    assert {t.__name__ for t in growth_agent.tools} == {
        "find_growth_opportunities", "draft_growth_actions",
    }
    # specialists may transfer between themselves (audit → growth handoff)
    assert audit_agent.disallow_transfer_to_peers is False
    assert growth_agent.disallow_transfer_to_peers is False


def test_sub_agents_have_routable_descriptions():
    """The coordinator routes on descriptions — they must exist and differ."""
    assert audit_agent.description and growth_agent.description
    assert audit_agent.description != growth_agent.description


# ==============================================================================
# TOOL 1 — generate_audit_prompts (writes session state)
# ==============================================================================
async def test_generate_audit_prompts_creates_session_state(monkeypatch, ctx):
    _mock_planning(monkeypatch)

    out = await generate_audit_prompts("sight360.com", "LASIK surgery", tool_context=ctx)

    audit_id = out["audit_id"]
    assert ctx.state["active_audit_id"] == audit_id
    stored = ctx.state[f"audit:{audit_id}"]
    assert stored["domain"] == "sight360.com"
    assert stored["location"] == "United States"      # default applied
    assert stored["responses"] == []
    assert stored["audit_log"] == []

    assert out["brand_name"] == "Sight360"
    assert out["industry"] == "LASIK & Vision Correction"
    assert out["known_competitors"] == IDENTITY["competitors"][:5]   # capped at 5
    assert out["prompts"] == [
        {"query": "Sight360 reviews 2026", "type": "branded"},
        {"query": "best lasik clinics philadelphia", "type": "unbranded"},
    ]
    assert "query_engines" in out["next_step"]


async def test_generate_audit_prompts_custom_location(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    out = await generate_audit_prompts("sight360.com", "LASIK", location="Germany", tool_context=ctx)
    assert ctx.state[f"audit:{out['audit_id']}"]["location"] == "Germany"
    kwargs = agent_mod.auto_generate_brand_prompts.await_args.kwargs
    assert kwargs["location"] == "Germany"


async def test_generate_audit_prompts_brand_name_fallback(monkeypatch, ctx):
    """Identity without brand_name → derive from domain."""
    monkeypatch.setattr(agent_mod, "analyze_brand_identity", AsyncMock(return_value={}))
    monkeypatch.setattr(agent_mod, "auto_generate_brand_prompts", AsyncMock(return_value=[dict(PROMPTS[0])]))
    out = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    assert out["brand_name"] == "Sight360"


async def test_concurrent_audits_isolated_in_state(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    a = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    b = await generate_audit_prompts("lasikplus.com", "LASIK", tool_context=ctx)
    assert a["audit_id"] != b["audit_id"]
    assert ctx.state[f"audit:{a['audit_id']}"]["domain"] == "sight360.com"
    assert ctx.state[f"audit:{b['audit_id']}"]["domain"] == "lasikplus.com"
    assert ctx.state["active_audit_id"] == b["audit_id"]   # most recent wins


# ==============================================================================
# audit_id fallback — the demo-robustness feature
# ==============================================================================
async def test_tools_fall_back_to_active_audit(monkeypatch, ctx):
    """Model forgets the id → tools use the session's active audit."""
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "x" * 30, "citations": []},
        "gemini": {"text": "y" * 30, "citations": []},
    }))

    out = await query_engines(tool_context=ctx)            # ← no audit_id passed
    assert out["audit_id"] == session["audit_id"]
    assert out["responses_collected"] == 4


async def test_explicit_audit_id_beats_active(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    a = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    await generate_audit_prompts("lasikplus.com", "LASIK", tool_context=ctx)  # becomes active
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "x" * 30, "citations": []},
        "gemini": {"text": "y" * 30, "citations": []},
    }))

    out = await query_engines(a["audit_id"], tool_context=ctx)   # explicit older audit
    assert out["audit_id"] == a["audit_id"]


async def test_no_audit_anywhere_errors(ctx):
    out = await query_engines(tool_context=ctx)            # empty state, no id
    assert "error" in out
    assert "generate_audit_prompts" in out["error"]


# ==============================================================================
# TOOL 2 — query_engines
# ==============================================================================
async def test_query_engines_unknown_audit_id(ctx):
    out = await query_engines("nonexistent", tool_context=ctx)
    assert "error" in out
    assert "generate_audit_prompts" in out["error"]


async def test_query_engines_happy_path(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    audit_id = session["audit_id"]

    long_text = "A" * 500
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": long_text, "citations": [{"url": "https://a.com", "title": "A"}]},
        "gemini": {"text": "short answer", "citations": []},
    }))

    out = await query_engines(audit_id, tool_context=ctx)

    assert out["responses_collected"] == 4          # 2 prompts × 2 engines
    assert len(out["previews"]) == 4
    assert all(len(p["preview"]) <= 200 for p in out["previews"])   # context-window guard
    openai_previews = [p for p in out["previews"] if p["engine"] == "openai"]
    assert openai_previews[0]["citations_found"] == 1
    # raw responses stored in session state, not returned
    stored = ctx.state[f"audit:{audit_id}"]
    assert len(stored["responses"]) == 4
    assert stored["responses"][0]["response"]["text"] == long_text
    assert "grade_responses" in out["next_step"]


async def test_query_engines_rerun_replaces_not_duplicates(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "x" * 30, "citations": []},
        "gemini": {"text": "y" * 30, "citations": []},
    }))
    await query_engines(session["audit_id"], tool_context=ctx)
    out = await query_engines(session["audit_id"], tool_context=ctx)   # re-run
    assert out["responses_collected"] == 4                              # not 8


async def test_query_engines_handles_none_text(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": None, "citations": []},
        "gemini": {"text": "ok", "citations": []},
    }))
    out = await query_engines(session["audit_id"], tool_context=ctx)
    previews = {p["engine"]: p["preview"] for p in out["previews"][:2]}
    assert previews["openai"] == ""                  # None → "" not a crash


# ==============================================================================
# TOOL 3 — grade_responses
# ==============================================================================
async def test_grade_responses_unknown_audit_id(ctx):
    out = await grade_responses("nonexistent", tool_context=ctx)
    assert "error" in out


async def test_grade_responses_before_query_engines(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    out = await grade_responses(session["audit_id"], tool_context=ctx)
    assert "error" in out
    assert "query_engines" in out["error"]


async def test_grade_responses_full_pipeline(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    audit_id = session["audit_id"]

    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "Sight360 is great " * 10, "citations": []},
        "gemini": {"text": "LasikPlus is better " * 10, "citations": []},
    }))
    await query_engines(audit_id, tool_context=ctx)

    visible = {
        "is_visible": True, "rank": 1, "sentiment": "Positive", "visibility_score": 100,
        "sentiment_score": 100, "cited_sources": [], "source_titles": {},
        "ranking_table": [{"rank": 1, "name": "Sight360", "sentiment": "Positive", "cited_url": None},
                          {"rank": 2, "name": "LasikPlus", "sentiment": "Neutral", "cited_url": None}],
    }
    invisible = {
        "is_visible": False, "rank": 0, "sentiment": "Neutral", "visibility_score": 0,
        "sentiment_score": 50, "cited_sources": [], "source_titles": {},
        "ranking_table": [{"rank": 1, "name": "LasikPlus", "sentiment": "Positive", "cited_url": None}],
    }
    monkeypatch.setattr(agent_mod, "grade_result", AsyncMock(side_effect=[visible, invisible, visible, invisible]))
    summary = {"positioning": "ok", "key_selling_points": [], "negative_risks": []}
    monkeypatch.setattr(agent_mod, "generate_executive_summary", AsyncMock(return_value=summary))

    out = await grade_responses(audit_id, tool_context=ctx)

    assert out["audit_id"] == audit_id
    lb = out["leaderboard"]
    assert lb["brand"]["name"] == "Sight360"
    assert lb["brand"]["mentions"] == 2
    assert lb["brand"]["visibility_rate"] == 0.5
    assert lb["responses_graded"] == 4
    comp_names = [c["name"] for c in lb["competitors"]]
    assert "LasikPlus" in comp_names
    assert "Sight360" not in comp_names
    assert len(out["visibility_gaps"]) == 2
    assert out["executive_summary"] == summary
    assert len(out["detail"]) == 4
    assert {d["visible"] for d in out["detail"]} == {True, False}
    # audit log persisted in session state for the growth agent
    assert len(ctx.state[f"audit:{audit_id}"]["audit_log"]) == 4


# ==============================================================================
# TOOL 4 — find_growth_opportunities
# ==============================================================================
REDDIT_URL = "https://reddit.com/r/lasik/comments/x1/thread"
BLOG_URL = "https://blog.example.com/lasik-guide"


def _seed_graded_audit(ctx, topic="LASIK"):
    """Plant a graded audit session directly in session state."""
    audit_id = "testaudit0001"
    ctx.state[f"audit:{audit_id}"] = {
        "domain": "sight360.com",
        "brand_name": "Sight360",
        "topic": topic,
        "location": "United States",
        "identity": dict(IDENTITY),
        "prompts": [dict(p) for p in PROMPTS],
        "responses": [{"prompt": "p", "prompt_type": "unbranded", "engine": "gemini", "response": {}}],
        "audit_log": [
            {"prompt": "best lasik philly", "prompt_type": "unbranded", "engine": "gemini",
             "grade": {"is_visible": False, "cited_sources": [REDDIT_URL, BLOG_URL], "source_titles": {}}},
            {"prompt": "best lasik philly", "prompt_type": "unbranded", "engine": "openai",
             "grade": {"is_visible": True, "cited_sources": [REDDIT_URL], "source_titles": {}}},
            {"prompt": "lasik cost", "prompt_type": "unbranded", "engine": "gemini",
             "grade": {"is_visible": True, "cited_sources": [REDDIT_URL], "source_titles": {}}},
        ],
    }
    ctx.state["active_audit_id"] = audit_id
    return audit_id


async def test_find_growth_unknown_audit_id(ctx):
    out = await find_growth_opportunities("nonexistent", tool_context=ctx)
    assert "error" in out


async def test_find_growth_before_grading(monkeypatch, ctx):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK", tool_context=ctx)
    out = await find_growth_opportunities(session["audit_id"], tool_context=ctx)
    assert "error" in out
    assert "grade_responses" in out["error"]


async def test_find_growth_no_community_sources(ctx):
    audit_id = _seed_graded_audit(ctx)
    for e in ctx.state[f"audit:{audit_id}"]["audit_log"]:
        e["grade"]["cited_sources"] = ["https://sight360.com/services"]
    out = await find_growth_opportunities(audit_id, tool_context=ctx)
    assert out["total_opportunities"] == 0
    assert "note" in out


async def test_find_growth_ranks_and_verifies(monkeypatch, ctx):
    audit_id = _seed_graded_audit(ctx)
    monkeypatch.setattr(agent_mod, "verify_urls", AsyncMock(return_value={
        REDDIT_URL: {"status": 200, "final_url": REDDIT_URL, "verdict": "verified"},
        BLOG_URL: {"status": 404, "final_url": BLOG_URL, "verdict": "hallucinated"},
    }))

    out = await find_growth_opportunities(audit_id, tool_context=ctx)

    assert out["total_opportunities"] == 2
    top = out["top_opportunities"]
    # reddit: 2 engines × 2 prompts → (50+10+20)×1.30 = 104.0; blog: (25+5+20)×0.50 = 25.0
    assert top[0]["url"] == REDDIT_URL
    assert top[0]["platform"] == "reddit"
    assert top[0]["score"] == 104.0
    assert top[0]["url_verdict"] == "verified"
    assert top[1]["score"] == 25.0
    assert top[1]["url_verdict"] == "hallucinated"
    assert out["platform_mix"] == {"reddit": 1, "blog": 1}
    assert out["url_verification"] == {"verified": 1, "hallucinated": 1}
    # opportunities persisted in session state for tool 5
    assert ctx.state[f"audit:{audit_id}"]["opportunities"][0]["url_verdict"] == "verified"


async def test_find_growth_falls_back_to_active(monkeypatch, ctx):
    """Growth agent can run without the id — session state carries it."""
    _seed_graded_audit(ctx)
    monkeypatch.setattr(agent_mod, "verify_urls", AsyncMock(return_value={
        REDDIT_URL: {"status": 200, "final_url": REDDIT_URL, "verdict": "verified"},
        BLOG_URL: {"status": 200, "final_url": BLOG_URL, "verdict": "verified"},
    }))
    out = await find_growth_opportunities(tool_context=ctx)     # ← no id
    assert out["total_opportunities"] == 2


# ==============================================================================
# TOOL 5 — draft_growth_actions
# ==============================================================================
async def test_draft_growth_unknown_audit_id(ctx):
    out = await draft_growth_actions("nonexistent", tool_context=ctx)
    assert "error" in out


async def test_draft_growth_before_mining(ctx):
    audit_id = _seed_graded_audit(ctx)
    out = await draft_growth_actions(audit_id, tool_context=ctx)
    assert "error" in out
    assert "find_growth_opportunities" in out["error"]


async def test_draft_growth_skips_hallucinated(monkeypatch, ctx):
    audit_id = _seed_graded_audit(ctx)
    monkeypatch.setattr(agent_mod, "verify_urls", AsyncMock(return_value={
        REDDIT_URL: {"status": 200, "final_url": REDDIT_URL, "verdict": "verified"},
        BLOG_URL: {"status": 404, "final_url": BLOG_URL, "verdict": "hallucinated"},
    }))
    await find_growth_opportunities(audit_id, tool_context=ctx)

    drafted = AsyncMock(return_value={
        "action_type": "community_reply", "headline": "h", "draft": "d",
        "why_this_source": "w", "effort": "low",
    })
    monkeypatch.setattr(agent_mod, "draft_growth_action", drafted)

    out = await draft_growth_actions(audit_id, top_n=5, tool_context=ctx)

    assert out["drafts_generated"] == 1            # hallucinated blog skipped
    assert out["drafts"][0]["url"] == REDDIT_URL
    assert out["drafts"][0]["url_verdict"] == "verified"
    assert out["plays"] == {"community_reply": 1}
    drafted.assert_awaited_once()                  # no LLM spend on fake URLs
    args = drafted.await_args.args
    assert args[1] == "Sight360" and args[2] == "sight360.com"
    # drafts persisted in session state
    assert len(ctx.state[f"audit:{audit_id}"]["growth_drafts"]) == 1


async def test_draft_growth_clamps_top_n(monkeypatch, ctx):
    audit_id = _seed_graded_audit(ctx)
    audit = ctx.state[f"audit:{audit_id}"]
    audit["opportunities"] = [
        {"source_url": f"https://reddit.com/r/x/comments/{i}/t", "platform": "reddit",
         "title": None, "engines": ["gemini"], "prompts": ["p"], "citation_count": 1,
         "prompt_breadth": 1, "score": 50.0, "url_verdict": "verified"}
        for i in range(12)
    ]
    monkeypatch.setattr(agent_mod, "draft_growth_action", AsyncMock(return_value={
        "action_type": "community_reply", "headline": "h", "draft": "d",
        "why_this_source": "w", "effort": "low",
    }))
    out = await draft_growth_actions(audit_id, top_n=99, tool_context=ctx)
    assert out["drafts_generated"] == 8            # hard cap at 8
