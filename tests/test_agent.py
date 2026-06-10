"""
tests/test_agent.py — ADK agent tool flow + audit session store.

sanbi_core functions are mocked at the agent-module level. Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import agents.sanbi_audit.agent as agent_mod
from agents.sanbi_audit.agent import (
    draft_growth_actions,
    find_growth_opportunities,
    generate_audit_prompts,
    grade_responses,
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


@pytest.fixture(autouse=True)
def clean_store():
    agent_mod._AUDITS.clear()
    yield
    agent_mod._AUDITS.clear()


def _mock_planning(monkeypatch):
    monkeypatch.setattr(agent_mod, "analyze_brand_identity", AsyncMock(return_value=dict(IDENTITY)))
    monkeypatch.setattr(agent_mod, "auto_generate_brand_prompts", AsyncMock(return_value=[dict(p) for p in PROMPTS]))


# ==============================================================================
# Root agent contract
# ==============================================================================
def test_root_agent_definition():
    assert root_agent.name == "sanbi_audit_agent"
    assert len(root_agent.tools) == 5
    tool_names = {t.__name__ for t in root_agent.tools}
    assert tool_names == {
        "generate_audit_prompts",
        "query_engines",
        "grade_responses",
        "find_growth_opportunities",
        "draft_growth_actions",
    }


# ==============================================================================
# TOOL 1 — generate_audit_prompts
# ==============================================================================
async def test_generate_audit_prompts_creates_session(monkeypatch):
    _mock_planning(monkeypatch)

    out = await generate_audit_prompts("sight360.com", "LASIK surgery")

    audit_id = out["audit_id"]
    assert audit_id in agent_mod._AUDITS
    assert out["brand_name"] == "Sight360"
    assert out["industry"] == "LASIK & Vision Correction"
    assert out["known_competitors"] == IDENTITY["competitors"][:5]   # capped at 5
    assert out["prompts"] == [
        {"query": "Sight360 reviews 2026", "type": "branded"},
        {"query": "best lasik clinics philadelphia", "type": "unbranded"},
    ]
    assert "query_engines" in out["next_step"]

    stored = agent_mod._AUDITS[audit_id]
    assert stored["domain"] == "sight360.com"
    assert stored["location"] == "United States"      # default applied
    assert stored["responses"] == []
    assert stored["audit_log"] == []


async def test_generate_audit_prompts_custom_location(monkeypatch):
    _mock_planning(monkeypatch)
    out = await generate_audit_prompts("sight360.com", "LASIK", location="Germany")
    assert agent_mod._AUDITS[out["audit_id"]]["location"] == "Germany"
    # location must be forwarded to prompt generation
    kwargs = agent_mod.auto_generate_brand_prompts.await_args.kwargs
    assert kwargs["location"] == "Germany"


async def test_generate_audit_prompts_brand_name_fallback(monkeypatch):
    """Identity without brand_name → derive from domain."""
    monkeypatch.setattr(agent_mod, "analyze_brand_identity", AsyncMock(return_value={}))
    monkeypatch.setattr(agent_mod, "auto_generate_brand_prompts", AsyncMock(return_value=[dict(PROMPTS[0])]))
    out = await generate_audit_prompts("sight360.com", "LASIK")
    assert out["brand_name"] == "Sight360"


async def test_concurrent_audits_are_isolated(monkeypatch):
    _mock_planning(monkeypatch)
    a = await generate_audit_prompts("sight360.com", "LASIK")
    b = await generate_audit_prompts("lasikplus.com", "LASIK")
    assert a["audit_id"] != b["audit_id"]
    assert agent_mod._AUDITS[a["audit_id"]]["domain"] == "sight360.com"
    assert agent_mod._AUDITS[b["audit_id"]]["domain"] == "lasikplus.com"


# ==============================================================================
# TOOL 2 — query_engines
# ==============================================================================
async def test_query_engines_unknown_audit_id():
    out = await query_engines("nonexistent")
    assert "error" in out
    assert "generate_audit_prompts" in out["error"]


async def test_query_engines_happy_path(monkeypatch):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK")
    audit_id = session["audit_id"]

    long_text = "A" * 500
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": long_text, "citations": [{"url": "https://a.com", "title": "A"}]},
        "gemini": {"text": "short answer", "citations": []},
    }))

    out = await query_engines(audit_id)

    assert out["responses_collected"] == 4          # 2 prompts × 2 engines
    assert len(out["previews"]) == 4
    assert all(len(p["preview"]) <= 200 for p in out["previews"])   # context-window guard
    openai_previews = [p for p in out["previews"] if p["engine"] == "openai"]
    assert openai_previews[0]["citations_found"] == 1
    # raw responses stored server-side, not returned
    assert len(agent_mod._AUDITS[audit_id]["responses"]) == 4
    assert agent_mod._AUDITS[audit_id]["responses"][0]["response"]["text"] == long_text
    assert "grade_responses" in out["next_step"]


async def test_query_engines_handles_none_text(monkeypatch):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK")
    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": None, "citations": []},
        "gemini": {"text": "ok", "citations": []},
    }))
    out = await query_engines(session["audit_id"])
    previews = {p["engine"]: p["preview"] for p in out["previews"][:2]}
    assert previews["openai"] == ""                  # None → "" not a crash


# ==============================================================================
# TOOL 3 — grade_responses
# ==============================================================================
async def test_grade_responses_unknown_audit_id():
    out = await grade_responses("nonexistent")
    assert "error" in out


async def test_grade_responses_before_query_engines(monkeypatch):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK")
    out = await grade_responses(session["audit_id"])
    assert "error" in out
    assert "query_engines" in out["error"]


async def test_grade_responses_full_pipeline(monkeypatch):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK")
    audit_id = session["audit_id"]

    monkeypatch.setattr(agent_mod, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "Sight360 is great " * 10, "citations": []},
        "gemini": {"text": "LasikPlus is better " * 10, "citations": []},
    }))
    await query_engines(audit_id)

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
    # 4 responses: alternate visible / invisible
    monkeypatch.setattr(agent_mod, "grade_result", AsyncMock(side_effect=[visible, invisible, visible, invisible]))
    summary = {"positioning": "ok", "key_selling_points": [], "negative_risks": []}
    monkeypatch.setattr(agent_mod, "generate_executive_summary", AsyncMock(return_value=summary))

    out = await grade_responses(audit_id)

    assert out["audit_id"] == audit_id
    lb = out["leaderboard"]
    assert lb["brand"]["name"] == "Sight360"
    assert lb["brand"]["mentions"] == 2
    assert lb["brand"]["visibility_rate"] == 0.5
    assert lb["responses_graded"] == 4
    # LasikPlus appears in all 4 ranking tables; Sight360 excluded from competitors
    comp_names = [c["name"] for c in lb["competitors"]]
    assert "LasikPlus" in comp_names
    assert "Sight360" not in comp_names
    # gaps = the 2 invisible responses
    assert len(out["visibility_gaps"]) == 2
    assert out["executive_summary"] == summary
    assert len(out["detail"]) == 4
    assert {d["visible"] for d in out["detail"]} == {True, False}
    # audit log persisted for future tools
    assert len(agent_mod._AUDITS[audit_id]["audit_log"]) == 4


# ==============================================================================
# TOOL 4 — find_growth_opportunities
# ==============================================================================
REDDIT_URL = "https://reddit.com/r/lasik/comments/x1/thread"
BLOG_URL = "https://blog.example.com/lasik-guide"


def _seed_graded_audit(monkeypatch_store_topic="LASIK"):
    """Plant a graded audit session directly in the store."""
    audit_id = "testaudit0001"
    agent_mod._AUDITS[audit_id] = {
        "domain": "sight360.com",
        "brand_name": "Sight360",
        "topic": monkeypatch_store_topic,
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
    return audit_id


async def test_find_growth_unknown_audit_id():
    out = await find_growth_opportunities("nonexistent")
    assert "error" in out


async def test_find_growth_before_grading(monkeypatch):
    _mock_planning(monkeypatch)
    session = await generate_audit_prompts("sight360.com", "LASIK")
    out = await find_growth_opportunities(session["audit_id"])
    assert "error" in out
    assert "grade_responses" in out["error"]


async def test_find_growth_no_community_sources():
    audit_id = _seed_graded_audit()
    # strip the citations → nothing classifiable
    for e in agent_mod._AUDITS[audit_id]["audit_log"]:
        e["grade"]["cited_sources"] = ["https://sight360.com/services"]
    out = await find_growth_opportunities(audit_id)
    assert out["total_opportunities"] == 0
    assert "note" in out


async def test_find_growth_ranks_and_verifies(monkeypatch):
    audit_id = _seed_graded_audit()
    monkeypatch.setattr(agent_mod, "verify_urls", AsyncMock(return_value={
        REDDIT_URL: {"status": 200, "final_url": REDDIT_URL, "verdict": "verified"},
        BLOG_URL: {"status": 404, "final_url": BLOG_URL, "verdict": "hallucinated"},
    }))

    out = await find_growth_opportunities(audit_id)

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
    # opportunities persisted for tool 5
    assert agent_mod._AUDITS[audit_id]["opportunities"][0]["url_verdict"] == "verified"


# ==============================================================================
# TOOL 5 — draft_growth_actions
# ==============================================================================
async def test_draft_growth_unknown_audit_id():
    out = await draft_growth_actions("nonexistent")
    assert "error" in out


async def test_draft_growth_before_mining():
    audit_id = _seed_graded_audit()
    out = await draft_growth_actions(audit_id)
    assert "error" in out
    assert "find_growth_opportunities" in out["error"]


async def test_draft_growth_skips_hallucinated(monkeypatch):
    audit_id = _seed_graded_audit()
    monkeypatch.setattr(agent_mod, "verify_urls", AsyncMock(return_value={
        REDDIT_URL: {"status": 200, "final_url": REDDIT_URL, "verdict": "verified"},
        BLOG_URL: {"status": 404, "final_url": BLOG_URL, "verdict": "hallucinated"},
    }))
    await find_growth_opportunities(audit_id)

    drafted = AsyncMock(return_value={
        "action_type": "community_reply", "headline": "h", "draft": "d",
        "why_this_source": "w", "effort": "low",
    })
    monkeypatch.setattr(agent_mod, "draft_growth_action", drafted)

    out = await draft_growth_actions(audit_id, top_n=5)

    assert out["drafts_generated"] == 1            # hallucinated blog skipped
    assert out["drafts"][0]["url"] == REDDIT_URL
    assert out["drafts"][0]["url_verdict"] == "verified"
    assert out["plays"] == {"community_reply": 1}
    drafted.assert_awaited_once()                  # no LLM spend on fake URLs
    # brand context forwarded
    args = drafted.await_args.args
    assert args[1] == "Sight360" and args[2] == "sight360.com"


async def test_draft_growth_clamps_top_n(monkeypatch):
    audit_id = _seed_graded_audit()
    # 12 verified opportunities planted directly
    agent_mod._AUDITS[audit_id]["opportunities"] = [
        {"source_url": f"https://reddit.com/r/x/comments/{i}/t", "platform": "reddit",
         "title": None, "engines": ["gemini"], "prompts": ["p"], "citation_count": 1,
         "prompt_breadth": 1, "score": 50.0, "url_verdict": "verified"}
        for i in range(12)
    ]
    monkeypatch.setattr(agent_mod, "draft_growth_action", AsyncMock(return_value={
        "action_type": "community_reply", "headline": "h", "draft": "d",
        "why_this_source": "w", "effort": "low",
    }))
    out = await draft_growth_actions(audit_id, top_n=99)
    assert out["drafts_generated"] == 8            # hard cap at 8
