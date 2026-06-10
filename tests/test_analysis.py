"""
tests/test_analysis.py — scoring formula, grading guards, citation merge, leaderboard.

Every LLM call is mocked. Zero API spend.
"""

from unittest.mock import AsyncMock

import pytest

import sanbi_core.analysis as analysis
from sanbi_core.analysis import (
    _empty_grade,
    _norm_brand,
    _safe_int,
    build_leaderboard,
    calculate_aeo_score,
    generate_executive_summary,
    grade_result,
)


# ==============================================================================
# calculate_aeo_score — the deterministic scoring matrix
# ==============================================================================
@pytest.mark.parametrize(
    "rank,sentiment,expected",
    [
        # Rank decay, positive sentiment (multiplier 1.0)
        (1, "Positive", 100),
        (2, "Positive", 85),
        (3, "Positive", 70),
        (5, "Positive", 70),
        (6, "Positive", 50),
        (10, "Positive", 50),
        (11, "Positive", 30),
        (99, "Positive", 30),
        # Not visible — always 0 regardless of sentiment
        (0, "Positive", 0),
        (-1, "Positive", 0),
        (-5, "Negative", 0),
        # Sentiment multipliers
        (1, "Negative", 25),
        (1, "Neutral", 85),
        (1, "Balanced", 85),
        (2, "Negative", 21),     # int(85 * 0.25)
        (2, "Neutral", 72),      # int(85 * 0.85)
        # Case-insensitivity
        (1, "NEGATIVE", 25),
        (1, "neutral", 85),
        # Unknown sentiment falls through to positive multiplier
        (1, "Enthusiastic", 100),
        (1, "", 100),
        # Compound strings
        (1, "Mostly negative", 25),
        (1, "Slightly negative tone", 25),
    ],
)
def test_calculate_aeo_score_matrix(rank, sentiment, expected):
    assert calculate_aeo_score(rank, sentiment) == expected


# ==============================================================================
# _safe_int / _norm_brand — LLM output coercion helpers
# ==============================================================================
@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 0),
        ("2", 2),
        (3, 3),
        (3.0, 3),
        ("3.7", 3),
        ("N/A", 0),
        ("", 0),
        ("first", 0),
        ([], 0),
        ({}, 0),
        (True, 1),
    ],
)
def test_safe_int(value, expected):
    assert _safe_int(value) == expected


def test_safe_int_custom_default():
    assert _safe_int("garbage", default=7) == 7


@pytest.mark.parametrize(
    "a,b,equal",
    [
        ("Sight 360", "sight360", True),
        ("Sight360", "SIGHT-360", True),
        ("Acme Inc.", "acme inc", True),
        ("Allegro MicroSystems", "allegro-microsystems", True),
        ("Acme", "Apex", False),
        ("", "", True),
    ],
)
def test_norm_brand(a, b, equal):
    assert (_norm_brand(a) == _norm_brand(b)) is equal


# ==============================================================================
# grade_result — cost guards (must NOT call the LLM grader)
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        "",                                     # empty
        "   ",                                  # whitespace only
        "short",                                # < 20 chars
        "Error: connection reset by peer",      # gemini engine failure
        "[OpenAI Error] rate limit exceeded",   # openai engine failure
        "OpenAI API key missing.",              # engine disabled (23 chars — passes length check!)
        "Unknown engine: perplexity",           # routing failure
    ],
)
async def test_grade_result_skips_error_responses(monkeypatch, text):
    grader = AsyncMock()
    monkeypatch.setattr(analysis, "robust_json_call", grader)

    out = await grade_result("best lasik", {"text": text, "citations": []}, "sight360.com")

    grader.assert_not_awaited()  # the money guard: no LLM call on garbage input
    assert out == _empty_grade()
    assert out["visibility_score"] == 0
    assert out["is_visible"] is False


async def test_grade_result_handles_missing_text_key(monkeypatch):
    grader = AsyncMock()
    monkeypatch.setattr(analysis, "robust_json_call", grader)
    out = await grade_result("q", {}, "sight360.com")
    grader.assert_not_awaited()
    assert out == _empty_grade()


# ==============================================================================
# grade_result — happy path + dirty LLM output normalization
# ==============================================================================
LONG_RESPONSE = "Sight360 is the top LASIK provider in Philadelphia, followed by LasikPlus." * 3


async def test_grade_result_happy_path(monkeypatch):
    grader = AsyncMock(
        return_value={
            "is_visible": True,
            "rank": 1,
            "sentiment": "Positive",
            "ranking_table": [
                {"rank": 1, "name": "Sight360", "sentiment": "Positive", "cited_url": "https://sight360.com"},
                {"rank": 2, "name": "LasikPlus", "sentiment": "Neutral", "cited_url": None},
            ],
            "cited_sources": ["https://newsite.com/review"],
            "source_titles": {"https://newsite.com/review": "Review"},
        }
    )
    monkeypatch.setattr(analysis, "robust_json_call", grader)

    api_citations = [{"url": "https://sight360.com", "title": "Official"}]
    out = await grade_result("best lasik in philly", {"text": LONG_RESPONSE, "citations": api_citations}, "sight360.com")

    assert out["is_visible"] is True
    assert out["rank"] == 1
    assert out["visibility_score"] == 100
    assert out["sentiment_score"] == 100
    # API citations come first, then new LLM-found URLs
    assert out["cited_sources"] == ["https://sight360.com", "https://newsite.com/review"]
    assert out["source_titles"]["https://sight360.com"] == "Official"
    assert len(out["ranking_table"]) == 2


async def test_grade_result_coerces_dirty_llm_values(monkeypatch):
    """LLM returns rank as None / strings — must not crash (production bug)."""
    grader = AsyncMock(
        return_value={
            "is_visible": True,
            "rank": None,                      # ← crashed int() before fix
            "sentiment": None,                 # ← crashed str.lower() before fix
            "ranking_table": [
                {"rank": "2", "name": "Acme", "sentiment": None},
                {"rank": "not-a-number", "name": None, "sentiment": "Positive"},
                "garbage-row-not-a-dict",      # ← must be skipped
            ],
            "cited_sources": [],
            "source_titles": {},
        }
    )
    monkeypatch.setattr(analysis, "robust_json_call", grader)

    out = await grade_result("q", {"text": LONG_RESPONSE, "citations": []}, "sight360.com")

    assert out["rank"] == 0
    assert out["sentiment"] == "Neutral"
    assert out["visibility_score"] == 0
    table = out["ranking_table"]
    assert len(table) == 2                     # garbage row dropped
    assert table[0]["rank"] == 2               # "2" coerced
    assert table[1]["rank"] == 2               # "not-a-number" → positional default (i=2)
    assert table[1]["name"] == "Unknown"       # None name → "Unknown"


async def test_grade_result_sentiment_case_insensitive(monkeypatch):
    grader = AsyncMock(
        return_value={"is_visible": True, "rank": 1, "sentiment": "positive",
                      "ranking_table": [], "cited_sources": [], "source_titles": {}}
    )
    monkeypatch.setattr(analysis, "robust_json_call", grader)
    out = await grade_result("q", {"text": LONG_RESPONSE, "citations": []}, "sight360.com")
    assert out["sentiment_score"] == 100       # was 50 before the case-insensitivity fix


async def test_grade_result_filters_google_redirect_urls(monkeypatch):
    grader = AsyncMock(
        return_value={
            "is_visible": False, "rank": 0, "sentiment": "Neutral", "ranking_table": [],
            "cited_sources": [
                "https://vertexaisearch.cloud.google.com/grounding-api-redirect/xyz",
                "https://google.com/grounding/abc",
                "https://legit-source.com/page",
            ],
            "source_titles": {},
        }
    )
    monkeypatch.setattr(analysis, "robust_json_call", grader)
    out = await grade_result("q", {"text": LONG_RESPONSE, "citations": []}, "sight360.com")
    assert out["cited_sources"] == ["https://legit-source.com/page"]


async def test_grade_result_regex_fallback_when_no_citations(monkeypatch):
    grader = AsyncMock(
        return_value={"is_visible": True, "rank": 3, "sentiment": "Neutral",
                      "ranking_table": [], "cited_sources": [], "source_titles": {}}
    )
    monkeypatch.setattr(analysis, "robust_json_call", grader)
    text = (
        "Check https://fallback-found.com/page. for details, but never "
        "https://vertexaisearch.cloud.google.com/redirect/zzz — that's internal. " + LONG_RESPONSE
    )
    out = await grade_result("q", {"text": text, "citations": []}, "sight360.com")
    assert "https://fallback-found.com/page" in out["cited_sources"]
    assert all("vertexaisearch" not in u for u in out["cited_sources"])


async def test_grade_result_uses_clean_brand_name(monkeypatch):
    """https://www.sight-360.com must grade brand 'Sight 360', not 'Https://Www'."""
    grader = AsyncMock(
        return_value={"is_visible": True, "rank": 1, "sentiment": "Positive",
                      "ranking_table": [], "cited_sources": [], "source_titles": {}}
    )
    monkeypatch.setattr(analysis, "robust_json_call", grader)
    await grade_result("q", {"text": LONG_RESPONSE, "citations": []}, "https://www.sight-360.com")
    prompt_sent = grader.await_args.args[0]
    assert "'Sight 360'" in prompt_sent
    assert "Https" not in prompt_sent


async def test_grade_result_grader_exception_returns_empty(monkeypatch):
    monkeypatch.setattr(analysis, "robust_json_call", AsyncMock(side_effect=RuntimeError("boom")))
    out = await grade_result("q", {"text": LONG_RESPONSE, "citations": []}, "sight360.com")
    assert out == _empty_grade()


# ==============================================================================
# build_leaderboard — aggregation edge cases
# ==============================================================================
def _entry(engine, visible, score, table=None, prompt="p1"):
    return {
        "prompt": prompt,
        "prompt_type": "unbranded",
        "engine": engine,
        "grade": {
            "is_visible": visible,
            "visibility_score": score,
            "ranking_table": table or [],
        },
    }


def test_leaderboard_empty_log_no_division_errors():
    out = build_leaderboard([], "Sight360")
    assert out["brand"]["avg_visibility_score"] == 0
    assert out["brand"]["visibility_rate"] == 0
    assert out["brand"]["mentions"] == 0
    assert out["competitors"] == []
    assert out["by_engine"] == {}
    assert out["responses_graded"] == 0
    assert out["prompts_audited"] == 0


def test_leaderboard_excludes_brand_from_competitors_normalized():
    """'sight360' / 'Sight-360' rows must NOT appear as competitors of 'Sight 360'."""
    log = [
        _entry("openai", True, 100, [
            {"rank": 1, "name": "sight360", "sentiment": "Positive"},
            {"rank": 2, "name": "Sight-360", "sentiment": "Positive"},
            {"rank": 3, "name": "LasikPlus", "sentiment": "Neutral"},
        ]),
    ]
    out = build_leaderboard(log, "Sight 360")
    names = [c["name"] for c in out["competitors"]]
    assert names == ["LasikPlus"]


def test_leaderboard_competitor_aggregation_and_rank_hygiene():
    log = [
        _entry("openai", False, 0, [
            {"rank": 1, "name": "Acme", "sentiment": "Positive"},
            {"rank": 0, "name": "Acme", "sentiment": "Neutral"},      # rank 0 → excluded from avg
        ], prompt="p1"),
        _entry("gemini", True, 85, [
            {"rank": "2", "name": "Acme", "sentiment": "Neutral"},    # string rank → coerced
            {"rank": None, "name": "Beta Corp", "sentiment": "Neutral"},
        ], prompt="p1"),
        _entry("openai", True, 100, [
            {"rank": 1, "name": "Beta Corp", "sentiment": "Positive"},
        ], prompt="p2"),
    ]
    out = build_leaderboard(log, "Sight360")

    acme = next(c for c in out["competitors"] if c["name"] == "Acme")
    assert acme["mentions"] == 3
    assert acme["avg_rank"] == 1.5            # (1 + 2) / 2 — the rank-0 row excluded

    beta = next(c for c in out["competitors"] if c["name"] == "Beta Corp")
    assert beta["mentions"] == 2
    assert beta["avg_rank"] == 1.0            # None rank excluded

    # share_of_voice: total mentions = 3 (acme) + 2 (beta) + 2 (brand visible) = 7
    assert acme["share_of_voice"] == round(3 / 7, 3)

    # brand block
    assert out["brand"]["mentions"] == 2
    assert out["brand"]["visibility_rate"] == round(2 / 3, 3)
    assert out["brand"]["avg_visibility_score"] == round((0 + 85 + 100) / 3, 1)

    # engines + prompt dedup
    assert out["by_engine"]["openai"]["visibility_rate"] == 0.5
    assert out["by_engine"]["gemini"]["visibility_rate"] == 1.0
    assert out["prompts_audited"] == 2
    assert out["responses_graded"] == 3


def test_leaderboard_caps_competitors_at_ten():
    table = [{"rank": i, "name": f"Comp{i}", "sentiment": "Neutral"} for i in range(1, 15)]
    # NB: grade_result caps tables at 5 rows, but build_leaderboard must defend independently
    log = [_entry("openai", False, 0, table)]
    out = build_leaderboard(log, "Sight360")
    assert len(out["competitors"]) == 10


def test_leaderboard_skips_empty_competitor_names():
    log = [_entry("openai", False, 0, [
        {"rank": 1, "name": "", "sentiment": "Neutral"},
        {"rank": 2, "name": None, "sentiment": "Neutral"},
        {"rank": 3, "name": "  ", "sentiment": "Neutral"},
        {"rank": 4, "name": "RealCo", "sentiment": "Neutral"},
    ])]
    out = build_leaderboard(log, "Sight360")
    assert [c["name"] for c in out["competitors"]] == ["RealCo"]


# ==============================================================================
# generate_executive_summary
# ==============================================================================
async def test_executive_summary_passthrough(monkeypatch):
    payload = {"positioning": "Strong", "key_selling_points": [], "negative_risks": []}
    monkeypatch.setattr(analysis, "robust_json_call", AsyncMock(return_value=payload))
    out = await generate_executive_summary("sight360.com", [])
    assert out == payload


async def test_executive_summary_fallback_on_error(monkeypatch):
    monkeypatch.setattr(analysis, "robust_json_call", AsyncMock(side_effect=RuntimeError("api down")))
    out = await generate_executive_summary("sight360.com", [])
    assert out["positioning"] == "Analysis pending."
    assert out["key_selling_points"] == []
    assert out["negative_risks"] == []
