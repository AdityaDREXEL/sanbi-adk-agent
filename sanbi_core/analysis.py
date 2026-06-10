"""
sanbi_core/analysis.py — Response grading + visibility scoring.

Adapted from production Sanbi `core/analysis.py`.

The grader is itself an LLM call (Vertex Gemini): it reads an engine's
response and determines whether the audited brand is visible, at what rank,
and with what sentiment. The weighted score formula is deterministic.
"""

import json
import logging
import re
from typing import Any, Dict, List

from .gemini import robust_json_call

logger = logging.getLogger(__name__)

# Engine-level failure texts that must never be sent to the (paid) LLM grader.
_ERROR_PREFIXES = ("Error:", "[OpenAI Error]", "OpenAI API key missing", "Unknown engine:")


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce LLM-returned rank values (None, "2", "N/A", 3.0) to int."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _norm_brand(name: str) -> str:
    """Normalize brand names for comparison: 'Sight 360' == 'sight360'."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def calculate_aeo_score(rank: int, sentiment: str) -> int:
    """
    Weighted Visibility Score (0-100) based on Rank and Sentiment.
    Formula: Base_Score(Rank) * Sentiment_Multiplier
    """
    # 1. Base Score (rank decay)
    if rank <= 0:
        return 0
    elif rank == 1:
        base_score = 100
    elif rank == 2:
        base_score = 85
    elif rank <= 5:
        base_score = 70
    elif rank <= 10:
        base_score = 50
    else:
        base_score = 30  # mentioned, but buried deep

    # 2. Sentiment multiplier
    s_lower = sentiment.lower()
    if "negative" in s_lower:
        multiplier = 0.25  # high visibility with bad sentiment is detrimental
    elif "neutral" in s_lower or "balanced" in s_lower:
        multiplier = 0.85
    else:
        multiplier = 1.0

    return int(base_score * multiplier)


async def grade_result(prompt: str, response_data: Dict[str, Any], brand_domain: str) -> Dict[str, Any]:
    """
    Analyze a single engine response:
      1. LLM determines visibility, rank, sentiment, and competitor leaderboard.
      2. Merge API citations with any new URLs the LLM finds in the text.
      3. Filter internal Google grounding redirect links.
      4. Calculate the weighted visibility score.
    """
    response_text = response_data.get("text", "")
    api_citations = response_data.get("citations", [])

    # Cost guard: skip grading if the engine returned nothing useful
    if not response_text or len(response_text.strip()) < 20 or response_text.strip().startswith(_ERROR_PREFIXES):
        logger.info(f"Skipping grading — empty or error response ({len(response_text)} chars)")
        return _empty_grade()

    clean_domain = brand_domain.replace("https://", "").replace("http://", "").replace("www.", "")
    brand_name = clean_domain.split(".")[0].replace("-", " ").strip().title()

    reasoning_prompt = f"""
    You are an AI Auditor analyzing a search engine response for the brand '{brand_name}'.

    PROMPT: "{prompt}"
    RESPONSE TO ANALYZE:
    "{response_text[:15000]}"

    TASK:
    1. Determine if '{brand_name}' is visible (recommended/listed/mentioned).
    2. Determine the rank (1 = first mention). If not present, rank is 0.
    3. Analyze sentiment (Positive/Neutral/Negative).
    4. Build a ranking table of ALL brands mentioned, in order of appearance.
    5. CITATIONS: I have provided known API citations below. ALSO extract any
       additional URLs mentioned in the text body missing from that list.

    KNOWN API CITATIONS:
    {json.dumps(api_citations)}
    """

    json_instruction = """
    Format the analysis into this strict JSON:
    {
      "is_visible": boolean,
      "rank": integer (0 if not found),
      "sentiment": "string",
      "ranking_table": [
         { "rank": 1, "name": "Brand Name", "sentiment": "Positive", "cited_url": "url or null" }
      ],
      "cited_sources": ["url1", "url2"],
      "source_titles": { "url1": "Page Title 1" }
    }
    """

    try:
        data = await robust_json_call(reasoning_prompt, json_instruction, use_search=False)

        # ---- Merge & deduplicate citations ----
        final_urls: List[str] = []
        final_titles: Dict[str, str] = {}

        # A. API citations first (highest quality / ground truth)
        for c in api_citations:
            u = c.get("url")
            t = c.get("title", "Source")
            if u and u not in final_urls:
                final_urls.append(u)
                final_titles[u] = t

        # B. LLM-extracted citations (only if new and clean)
        for u in data.get("cited_sources", []):
            if "grounding-api-redirect" in u or "google.com/grounding" in u:
                continue
            if "vertexaisearch.cloud.google.com" in u:
                continue
            if u not in final_urls:
                final_urls.append(u)
                final_titles[u] = data.get("source_titles", {}).get(u, "Mentioned in Text")

        # C. Regex fallback if both failed
        if not final_urls:
            found = re.findall(r'https?://[^\s)\]"]+', response_text)
            cleaned = []
            for u in found:
                u = u.strip(".,)")
                if "grounding-api-redirect" in u or "vertexaisearch.cloud.google.com" in u:
                    continue
                cleaned.append(u)
            final_urls = list(set(cleaned))[:10]

        data["cited_sources"] = final_urls
        data["source_titles"] = final_titles

        # Normalize + score
        rank = _safe_int(data.get("rank", 0))
        sentiment = str(data.get("sentiment") or "Neutral")
        s_low = sentiment.lower()
        data["visibility_score"] = calculate_aeo_score(rank, sentiment)
        data["sentiment_score"] = 100 if "positive" in s_low else (0 if "negative" in s_low else 50)
        data["rank"] = rank
        data["sentiment"] = sentiment
        data["is_visible"] = bool(data.get("is_visible", False))

        normalized_table = []
        for i, row in enumerate(data.get("ranking_table", [])[:5], start=1):
            if isinstance(row, dict):
                normalized_table.append({
                    "rank": _safe_int(row.get("rank", i), i),
                    "name": str(row.get("name") or "Unknown"),
                    "sentiment": str(row.get("sentiment") or "Neutral"),
                    "cited_url": row.get("cited_url", None),
                })
        data["ranking_table"] = normalized_table

        return data

    except Exception as e:
        logger.error(f"Grading failed: {e}")
        return _empty_grade()


def _empty_grade() -> Dict[str, Any]:
    return {
        "is_visible": False,
        "rank": 0,
        "sentiment": "Neutral",
        "sentiment_score": 50,
        "visibility_score": 0,
        "ranking_table": [],
        "cited_sources": [],
        "source_titles": {},
    }


def build_leaderboard(audit_log: List[Dict[str, Any]], brand_name: str) -> Dict[str, Any]:
    """
    Aggregate per-prompt, per-engine grades into a competitive leaderboard.

    Returns:
      {
        "brand": {"name", "avg_visibility_score", "visibility_rate", "mentions"},
        "competitors": [{"name", "mentions", "avg_rank", "share_of_voice"}],
        "by_engine": {engine: {"visibility_rate", "avg_score"}},
      }
    """
    brand_scores: List[int] = []
    brand_visible = 0
    total_graded = 0
    competitor_stats: Dict[str, Dict[str, Any]] = {}
    engine_stats: Dict[str, Dict[str, List]] = {}

    for entry in audit_log:
        grade = entry.get("grade", {})
        engine = entry.get("engine", "unknown")
        total_graded += 1

        es = engine_stats.setdefault(engine, {"scores": [], "visible": []})
        es["scores"].append(grade.get("visibility_score", 0))
        es["visible"].append(1 if grade.get("is_visible") else 0)

        brand_scores.append(grade.get("visibility_score", 0))
        if grade.get("is_visible"):
            brand_visible += 1

        for row in grade.get("ranking_table", []):
            name = (row.get("name") or "").strip()
            if not name or _norm_brand(name) == _norm_brand(brand_name):
                continue
            cs = competitor_stats.setdefault(name, {"mentions": 0, "ranks": []})
            cs["mentions"] += 1
            r = _safe_int(row.get("rank", 0))
            if r > 0:
                cs["ranks"].append(r)

    total_mentions = sum(c["mentions"] for c in competitor_stats.values()) + brand_visible

    competitors = sorted(
        (
            {
                "name": name,
                "mentions": s["mentions"],
                "avg_rank": round(sum(s["ranks"]) / len(s["ranks"]), 1) if s["ranks"] else None,
                "share_of_voice": round(s["mentions"] / total_mentions, 3) if total_mentions else 0,
            }
            for name, s in competitor_stats.items()
        ),
        key=lambda x: x["mentions"],
        reverse=True,
    )[:10]

    by_engine = {
        e: {
            "visibility_rate": round(sum(s["visible"]) / len(s["visible"]), 3) if s["visible"] else 0,
            "avg_score": round(sum(s["scores"]) / len(s["scores"]), 1) if s["scores"] else 0,
        }
        for e, s in engine_stats.items()
    }

    return {
        "brand": {
            "name": brand_name,
            "avg_visibility_score": round(sum(brand_scores) / len(brand_scores), 1) if brand_scores else 0,
            "visibility_rate": round(brand_visible / total_graded, 3) if total_graded else 0,
            "mentions": brand_visible,
        },
        "competitors": competitors,
        "by_engine": by_engine,
        "prompts_audited": len({e.get("prompt") for e in audit_log}),
        "responses_graded": total_graded,
    }


async def generate_executive_summary(domain: str, audit_log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Executive summary (strengths/weaknesses) based on the full audit run."""
    wins = [x["prompt"] for x in audit_log if x.get("grade", {}).get("is_visible")]
    losses = [x["prompt"] for x in audit_log if not x.get("grade", {}).get("is_visible")]

    reasoning_prompt = f"""
    You are a Brand Reputation Analyst.
    Domain: {domain}

    DATA SUMMARY:
    - Visible Topics (Wins): {str(wins[:10])}
    - Invisible Topics (Losses): {str(losses[:10])}

    TASK:
    1. Define the brand's current AI-visibility positioning.
    2. Identify 2 key strengths.
    3. Identify 2 risks or content gaps.
    """

    json_instruction = """
    Format into this JSON:
    {
      "positioning": "2 sentence summary",
      "key_selling_points": [ { "title": "Strength", "description": "Why" } ],
      "negative_risks": [ { "title": "Risk", "description": "Why" } ]
    }
    """

    try:
        return await robust_json_call(reasoning_prompt, json_instruction, use_search=False)
    except Exception as e:
        logger.error(f"Executive summary failed: {e}")
        return {"positioning": "Analysis pending.", "key_selling_points": [], "negative_risks": []}
