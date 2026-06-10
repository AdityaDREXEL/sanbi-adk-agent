"""
tests/test_mcp_server.py — MCP tool registration + end-to-end audit flow (mocked).

Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import mcp_server.server as srv

IDENTITY = {
    "brand_name": "Sight360",
    "industry": "LASIK & Vision Correction",
    "target_audience": "Adults considering vision correction",
    "competitors": ["LasikPlus", "TLC"],
}

PROMPTS = [
    {"search_query": "Sight360 reviews 2026", "topic": "LASIK", "prompt_type": "branded", "location": "United States"},
    {"search_query": "best lasik clinics philadelphia", "topic": "LASIK", "prompt_type": "unbranded", "location": "United States"},
]

VISIBLE_GRADE = {
    "is_visible": True, "rank": 1, "sentiment": "Positive", "visibility_score": 100,
    "sentiment_score": 100, "cited_sources": [], "source_titles": {},
    "ranking_table": [{"rank": 2, "name": "LasikPlus", "sentiment": "Neutral", "cited_url": None}],
}
INVISIBLE_GRADE = {
    "is_visible": False, "rank": 0, "sentiment": "Neutral", "visibility_score": 0,
    "sentiment_score": 50, "cited_sources": [], "source_titles": {},
    "ranking_table": [{"rank": 1, "name": "LasikPlus", "sentiment": "Positive", "cited_url": None}],
}


async def test_tool_is_registered():
    tools = await srv.mcp.list_tools()
    names = [t.name for t in tools]
    assert "run_visibility_audit" in names
    tool = next(t for t in tools if t.name == "run_visibility_audit")
    # schema must expose the three documented params
    props = tool.inputSchema["properties"]
    assert set(props) == {"domain", "topic", "location"}
    assert tool.inputSchema.get("required") == ["domain"]


def _mock_pipeline(monkeypatch, identity=None):
    monkeypatch.setattr(srv, "analyze_brand_identity", AsyncMock(return_value=dict(identity or IDENTITY)))
    monkeypatch.setattr(srv, "auto_generate_brand_prompts", AsyncMock(return_value=[dict(p) for p in PROMPTS]))
    monkeypatch.setattr(srv, "query_all_engines", AsyncMock(return_value={
        "openai": {"text": "Sight360 is great " * 10, "citations": []},
        "gemini": {"text": "LasikPlus only " * 10, "citations": []},
    }))
    monkeypatch.setattr(srv, "grade_result", AsyncMock(
        side_effect=[VISIBLE_GRADE, INVISIBLE_GRADE, VISIBLE_GRADE, INVISIBLE_GRADE]))
    monkeypatch.setattr(srv, "generate_executive_summary", AsyncMock(
        return_value={"positioning": "ok", "key_selling_points": [], "negative_risks": []}))


async def test_run_visibility_audit_full_flow(monkeypatch):
    _mock_pipeline(monkeypatch)

    out = await srv.run_visibility_audit("sight360.com", "LASIK", "United States")

    assert out["brand"] == "Sight360"
    assert out["domain"] == "sight360.com"
    assert out["topic"] == "LASIK"
    assert out["prompts_audited"] == ["Sight360 reviews 2026", "best lasik clinics philadelphia"]
    assert out["identity"]["industry"] == "LASIK & Vision Correction"

    lb = out["leaderboard"]
    assert lb["brand"]["mentions"] == 2
    assert lb["responses_graded"] == 4
    assert [c["name"] for c in lb["competitors"]] == ["LasikPlus"]

    assert len(out["visibility_gaps"]) == 2
    assert out["executive_summary"]["positioning"] == "ok"


async def test_empty_topic_inferred_from_industry(monkeypatch):
    _mock_pipeline(monkeypatch)
    out = await srv.run_visibility_audit("sight360.com")          # no topic
    assert out["topic"] == "LASIK & Vision Correction"            # inferred from identity
    kwargs = srv.auto_generate_brand_prompts.await_args.kwargs
    assert kwargs["user_topic"] == "LASIK & Vision Correction"


async def test_empty_topic_and_empty_industry_uses_general(monkeypatch):
    identity = dict(IDENTITY, industry="")
    _mock_pipeline(monkeypatch, identity=identity)
    out = await srv.run_visibility_audit("sight360.com")
    assert out["topic"] == "general"


async def test_brand_name_fallback_from_domain(monkeypatch):
    _mock_pipeline(monkeypatch, identity={})                      # identity pipeline failed
    out = await srv.run_visibility_audit("sight360.com", "LASIK")
    assert out["brand"] == "Sight360"
