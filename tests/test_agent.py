"""
tests/test_agent.py — ADK agent tool flow + audit session store.

sanbi_core functions are mocked at the agent-module level. Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import agents.sanbi_audit.agent as agent_mod
from agents.sanbi_audit.agent import (
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
    assert len(root_agent.tools) == 3
    tool_names = {t.__name__ for t in root_agent.tools}
    assert tool_names == {"generate_audit_prompts", "query_engines", "grade_responses"}


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
