"""
tests/test_planning.py — identity pipeline + prompt generation.

Every LLM call is mocked. Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import sanbi_core.planning as planning
from sanbi_core.planning import (
    _fallback_identity,
    _validate_identity,
    analyze_brand_identity,
    auto_generate_brand_prompts,
)

IDENTITY = {
    "brand_name": "Sight360",
    "industry": "LASIK & Vision Correction Surgery",
    "specialty": "Bladeless LASIK",
    "business_model": "Fee-for-service eye surgery centers",
    "target_audience": "Adults 25-50 considering vision correction",
    "audience_pain_points": ["glasses inconvenience", "contact lens infections"],
    "competitors": ["LasikPlus", "TLC Laser Eye Centers"],
    "product_categories": ["LASIK", "PRK", "Cataract surgery"],
    "key_search_terms": ["lasik near me", "lasik cost 2026"],
}


# ==============================================================================
# _fallback_identity / _validate_identity
# ==============================================================================
@pytest.mark.parametrize(
    "domain,expected_name",
    [
        ("sight360.com", "Sight360"),
        ("https://www.sight360.com", "Sight360"),
        ("http://sight360.com", "Sight360"),
        ("www.allegro-micro.com", "Allegro-micro"),
        ("SUB.example.co.uk", "Sub"),
    ],
)
def test_fallback_identity_domain_cleaning(domain, expected_name):
    out = _fallback_identity(domain)
    assert out["brand_name"] == expected_name
    assert out["competitors"] == []
    assert out["key_search_terms"] == []


def test_validate_identity_fills_missing_keys():
    out = _validate_identity({"brand_name": "Acme"}, "acme.com")
    for key in ("industry", "specialty", "business_model", "target_audience",
                "audience_pain_points", "competitors", "product_categories", "key_search_terms"):
        assert key in out
    assert out["brand_name"] == "Acme"
    assert out["competitors"] == []


def test_validate_identity_replaces_falsy_values():
    out = _validate_identity({"brand_name": None, "industry": "", "competitors": None}, "acme.com")
    assert out["brand_name"] == "Acme"        # derived from domain
    assert out["industry"] == ""
    assert out["competitors"] == []


def test_validate_identity_preserves_good_values():
    out = _validate_identity(dict(IDENTITY), "sight360.com")
    assert out["industry"] == "LASIK & Vision Correction Surgery"
    assert out["competitors"] == ["LasikPlus", "TLC Laser Eye Centers"]


def test_validate_identity_generic_industry_warns_not_crashes():
    out = _validate_identity({"brand_name": "X", "industry": "Technology"}, "x.com")
    assert out["industry"] == "Technology"    # warned, not mutated


# ==============================================================================
# analyze_brand_identity — pipeline branches
# ==============================================================================
async def test_identity_empty_research_falls_back(monkeypatch):
    monkeypatch.setattr(planning, "gemini_grounded_text", AsyncMock(return_value=("", [])))
    extract = AsyncMock()
    monkeypatch.setattr(planning, "gemini_force_json", extract)

    out = await analyze_brand_identity("sight360.com")
    extract.assert_not_awaited()              # never pay for extraction without research
    assert out["brand_name"] == "Sight360"
    assert out["industry"] == ""


async def test_identity_short_research_falls_back(monkeypatch):
    monkeypatch.setattr(planning, "gemini_grounded_text", AsyncMock(return_value=("too short", [])))
    out = await analyze_brand_identity("sight360.com")
    assert out == _fallback_identity("sight360.com")


async def test_identity_happy_path(monkeypatch):
    research = "Sight360 is a Florida-based eye care group offering LASIK..." * 5
    monkeypatch.setattr(planning, "gemini_grounded_text", AsyncMock(return_value=(research, [])))
    monkeypatch.setattr(planning, "gemini_force_json", AsyncMock(return_value=dict(IDENTITY)))

    out = await analyze_brand_identity("sight360.com")
    assert out["brand_name"] == "Sight360"
    assert out["industry"] == "LASIK & Vision Correction Surgery"
    # validation ran: all keys present
    assert "audience_pain_points" in out


async def test_identity_extraction_empty_falls_back(monkeypatch):
    research = "Long enough research text about the company. " * 5
    monkeypatch.setattr(planning, "gemini_grounded_text", AsyncMock(return_value=(research, [])))
    monkeypatch.setattr(planning, "gemini_force_json", AsyncMock(return_value={}))
    out = await analyze_brand_identity("sight360.com")
    assert out == _fallback_identity("sight360.com")


async def test_identity_extraction_exception_falls_back(monkeypatch):
    research = "Long enough research text about the company. " * 5
    monkeypatch.setattr(planning, "gemini_grounded_text", AsyncMock(return_value=(research, [])))
    monkeypatch.setattr(planning, "gemini_force_json", AsyncMock(side_effect=RuntimeError("parse fail")))
    out = await analyze_brand_identity("sight360.com")
    assert out == _fallback_identity("sight360.com")


async def test_identity_url_prefix_cleaned_in_research_prompt(monkeypatch):
    research_mock = AsyncMock(return_value=("", []))
    monkeypatch.setattr(planning, "gemini_grounded_text", research_mock)
    await analyze_brand_identity("https://www.sight360.com")
    prompt_sent = research_mock.await_args.args[0]
    assert '"sight360.com"' in prompt_sent
    assert "https://www" not in prompt_sent


# ==============================================================================
# auto_generate_brand_prompts — filtering, typing, fallbacks
# ==============================================================================
def _llm_prompts(*texts_types):
    return {"prompts": [{"text": t, "type": ty} for t, ty in texts_types]}


async def test_prompts_happy_path_with_ground_truth_typing(monkeypatch):
    """prompt_type must come from text content, not the LLM's (unreliable) label."""
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(return_value=_llm_prompts(
        ("Sight360 LASIK reviews 2026", "unbranded"),                  # mislabeled → branded
        ("best lasik surgeons in philadelphia 2026", "branded"),       # mislabeled → unbranded
        ("is sight360 worth it for PRK", "unbranded"),                 # domain stem → branded
        ("how much does laser eye surgery cost", "unbranded"),
    )))

    out = await auto_generate_brand_prompts("LASIK", "sight360.com", identity=dict(IDENTITY))

    by_text = {p["search_query"]: p["prompt_type"] for p in out}
    assert by_text["Sight360 LASIK reviews 2026"] == "branded"
    assert by_text["best lasik surgeons in philadelphia 2026"] == "unbranded"
    assert by_text["is sight360 worth it for PRK"] == "branded"
    assert by_text["how much does laser eye surgery cost"] == "unbranded"
    assert all(p["location"] == "United States" for p in out)
    assert all(p["topic"] == "LASIK" for p in out)


async def test_prompts_filters_generic_fillers_and_short_text(monkeypatch):
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(return_value=_llm_prompts(
        ("best online services for vision", "unbranded"),      # filler: "online services"
        ("top technology company for eyes", "unbranded"),      # filler: "technology company"
        ("short", "unbranded"),                                # < 10 chars
        ("", "unbranded"),                                     # empty
        ("legitimate lasik recovery time question", "unbranded"),
    )))
    out = await auto_generate_brand_prompts("LASIK", "sight360.com", identity=dict(IDENTITY))
    assert [p["search_query"] for p in out] == ["legitimate lasik recovery time question"]


async def test_prompts_dedups_existing_queries(monkeypatch):
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(return_value=_llm_prompts(
        ("best lasik surgeons in philadelphia", "unbranded"),
        ("BEST LASIK SURGEONS IN PHILADELPHIA", "unbranded"),  # case-dup of the first
        ("lasik cost comparison guide 2026", "unbranded"),
    )))
    out = await auto_generate_brand_prompts(
        "LASIK", "sight360.com",
        existing_queries=["Best LASIK surgeons in Philadelphia"],
        identity=dict(IDENTITY),
    )
    assert [p["search_query"] for p in out] == ["lasik cost comparison guide 2026"]


async def test_prompts_respects_max_count(monkeypatch):
    many = [(f"unique lasik question number {i} explained", "unbranded") for i in range(20)]
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(return_value=_llm_prompts(*many)))
    out = await auto_generate_brand_prompts("LASIK", "sight360.com", max_count=4, identity=dict(IDENTITY))
    assert len(out) == 4


async def test_prompts_empty_llm_response_returns_fallback(monkeypatch):
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(return_value={}))
    out = await auto_generate_brand_prompts("LASIK", "sight360.com", identity=dict(IDENTITY))
    assert len(out) == 1
    assert out[0]["prompt_type"] == "branded"
    assert "Sight360" in out[0]["search_query"]


async def test_prompts_exception_returns_fallback(monkeypatch):
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(side_effect=RuntimeError("api down")))
    out = await auto_generate_brand_prompts("LASIK", "sight360.com", identity=dict(IDENTITY))
    assert len(out) == 1
    assert out[0]["prompt_type"] == "branded"
    assert "reviews" in out[0]["search_query"]


async def test_prompts_skips_research_when_identity_provided(monkeypatch):
    research = AsyncMock()
    monkeypatch.setattr(planning, "analyze_brand_identity", research)
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(return_value=_llm_prompts(
        ("some valid lasik question here", "unbranded"),
    )))
    await auto_generate_brand_prompts("LASIK", "sight360.com", identity=dict(IDENTITY))
    research.assert_not_awaited()             # the ADK agent researches once, reuses everywhere


async def test_prompts_researches_when_identity_missing(monkeypatch):
    research = AsyncMock(return_value=dict(IDENTITY))
    monkeypatch.setattr(planning, "analyze_brand_identity", research)
    monkeypatch.setattr(planning, "robust_json_call", AsyncMock(return_value=_llm_prompts(
        ("some valid lasik question here", "unbranded"),
    )))
    await auto_generate_brand_prompts("LASIK", "sight360.com")
    research.assert_awaited_once()
