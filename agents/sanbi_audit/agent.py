"""
agents/sanbi_audit/agent.py — SanbiAuditAgent (ADK root agent).

A single ADK agent with five tools that walk the full Sanbi pipeline —
measure visibility, then act on it:

    1. generate_audit_prompts      → researches the brand, plans audit queries
    2. query_engines               → runs every prompt across OpenAI + Vertex Gemini
    3. grade_responses             → LLM-grades each response, builds leaderboard
    4. find_growth_opportunities   → classifies + ranks + verifies cited sources
    5. draft_growth_actions        → platform-branched growth content per source

Design note: raw engine responses are large (5-15k chars each). Tools share
them through an in-process audit store keyed by audit_id instead of routing
them through the agent's context window. Each tool returns compact JSON the
agent can reason over and narrate in the ADK dev UI.

Run locally:
    adk web agents          # from repo root → http://localhost:8000
"""

import logging
import uuid
from typing import Optional

from google.adk.agents import Agent

from sanbi_core.planning import analyze_brand_identity, auto_generate_brand_prompts
from sanbi_core.execution import query_all_engines, ENGINES
from sanbi_core.analysis import grade_result, build_leaderboard, generate_executive_summary
from sanbi_core.growth import build_opportunities, draft_growth_action
from sanbi_core.verifier import verify_urls

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# In-process audit session store.
# Cloud Run demo scale = 1 instance, so module state is fine. The production
# refactor swaps this for AlloyDB / Memorystore (see README roadmap).
# ------------------------------------------------------------------------------
_AUDITS: dict[str, dict] = {}


# ==============================================================================
# TOOL 1 — Plan the audit
# ==============================================================================
async def generate_audit_prompts(brand_domain: str, topic: str, location: Optional[str] = None) -> dict:
    """Research a brand and generate the audit prompt plan.

    Researches the company behind `brand_domain` using Google-Search-grounded
    Gemini (identity: industry, audience, competitors, products), then
    generates branded + unbranded search queries that real customers would
    type into AI assistants.

    Args:
        brand_domain: The brand's website domain, e.g. "sight360.com".
        topic: Audit focus area, e.g. "LASIK surgery" or "current sensors".
        location: Market to simulate, e.g. "United States" (default).

    Returns:
        dict with audit_id (pass to the other tools), brand identity summary,
        and the planned prompts.
    """
    loc = location or "United States"
    identity = await analyze_brand_identity(brand_domain)
    prompts = await auto_generate_brand_prompts(
        user_topic=topic,
        domain=brand_domain,
        location=loc,
        max_count=6,
        branded_count=2,
        unbranded_count=4,
        identity=identity,
    )

    audit_id = uuid.uuid4().hex[:12]
    _AUDITS[audit_id] = {
        "domain": brand_domain,
        "brand_name": identity.get("brand_name", brand_domain.split(".")[0].title()),
        "topic": topic,
        "location": loc,
        "identity": identity,
        "prompts": prompts,
        "responses": [],   # filled by query_engines
        "audit_log": [],   # filled by grade_responses
    }

    return {
        "audit_id": audit_id,
        "brand_name": _AUDITS[audit_id]["brand_name"],
        "industry": identity.get("industry"),
        "target_audience": identity.get("target_audience"),
        "known_competitors": identity.get("competitors", [])[:5],
        "prompts": [
            {"query": p["search_query"], "type": p["prompt_type"]} for p in prompts
        ],
        "next_step": "Call query_engines with this audit_id.",
    }


# ==============================================================================
# TOOL 2 — Query the engines
# ==============================================================================
async def query_engines(audit_id: str) -> dict:
    """Run every planned prompt across all AI engines (OpenAI + Vertex Gemini).

    Each engine answers independently using its native capabilities (Gemini
    uses live Google Search grounding; OpenAI answers from training data).
    Raw responses are stored server-side; this returns compact previews.

    Args:
        audit_id: The audit session id returned by generate_audit_prompts.

    Returns:
        dict with per-prompt, per-engine response previews and citation counts.
    """
    audit = _AUDITS.get(audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id}'. Call generate_audit_prompts first."}

    previews = []
    for p in audit["prompts"]:
        prompt_text = p["search_query"]
        engine_results = await query_all_engines(prompt_text, audit["location"])
        for engine, res in engine_results.items():
            audit["responses"].append({
                "prompt": prompt_text,
                "prompt_type": p["prompt_type"],
                "engine": engine,
                "response": res,
            })
            previews.append({
                "prompt": prompt_text,
                "engine": engine,
                "preview": (res.get("text") or "")[:200],
                "citations_found": len(res.get("citations", [])),
            })

    return {
        "audit_id": audit_id,
        "engines": ENGINES,
        "responses_collected": len(audit["responses"]),
        "previews": previews,
        "next_step": "Call grade_responses with this audit_id.",
    }


# ==============================================================================
# TOOL 3 — Grade + leaderboard
# ==============================================================================
async def grade_responses(audit_id: str) -> dict:
    """Grade every engine response and build the competitive leaderboard.

    For each stored response, an LLM grader determines: is the brand visible,
    at what rank, with what sentiment, and which competitors appear. Scores
    use Sanbi's weighted formula (rank decay x sentiment multiplier). Returns
    the leaderboard, per-engine breakdown, and executive summary.

    Args:
        audit_id: The audit session id returned by generate_audit_prompts.

    Returns:
        dict with brand score, competitor leaderboard, per-engine stats,
        gap analysis (prompts where the brand was invisible), and summary.
    """
    audit = _AUDITS.get(audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id}'. Call generate_audit_prompts first."}
    if not audit["responses"]:
        return {"error": "No responses collected yet. Call query_engines first."}

    audit_log = []
    for entry in audit["responses"]:
        grade = await grade_result(entry["prompt"], entry["response"], audit["domain"])
        audit_log.append({
            "prompt": entry["prompt"],
            "prompt_type": entry["prompt_type"],
            "engine": entry["engine"],
            "grade": grade,
        })
    audit["audit_log"] = audit_log

    leaderboard = build_leaderboard(audit_log, audit["brand_name"])
    summary = await generate_executive_summary(audit["domain"], audit_log)

    gaps = [
        {"prompt": e["prompt"], "engine": e["engine"], "type": e["prompt_type"]}
        for e in audit_log
        if not e["grade"].get("is_visible")
    ]

    return {
        "audit_id": audit_id,
        "leaderboard": leaderboard,
        "visibility_gaps": gaps[:10],
        "executive_summary": summary,
        "detail": [
            {
                "prompt": e["prompt"],
                "engine": e["engine"],
                "visible": e["grade"].get("is_visible"),
                "rank": e["grade"].get("rank"),
                "sentiment": e["grade"].get("sentiment"),
                "score": e["grade"].get("visibility_score"),
            }
            for e in audit_log
        ],
    }


# ==============================================================================
# TOOL 4 — Growth opportunity mining (classify → rank → verify)
# ==============================================================================
async def find_growth_opportunities(audit_id: str) -> dict:
    """Find, rank, and verify the community sources AI engines cited.

    Runs every citation from the graded audit through Sanbi's deterministic
    platform classifier (reddit/forum/qa/youtube/reviews/blog/wiki/...),
    aggregates citations per URL, and ranks them with the production scoring
    formula — (engines x 25 + prompt_breadth x 5 + recency x 20) x platform
    replyability weight. Then verifies the top URLs are REAL (HEAD checks,
    YouTube oEmbed) — AI engines sometimes hallucinate citations.

    Args:
        audit_id: The audit session id returned by generate_audit_prompts.

    Returns:
        dict with platform mix, hallucination check results, and the ranked
        top opportunities (each with url, platform, score, url_verdict).
    """
    audit = _AUDITS.get(audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id}'. Call generate_audit_prompts first."}
    if not audit["audit_log"]:
        return {"error": "Audit not graded yet. Call grade_responses first."}

    opportunities = build_opportunities(audit["audit_log"])
    if not opportunities:
        return {
            "audit_id": audit_id,
            "total_opportunities": 0,
            "note": "No community/growth surfaces were cited in this audit.",
        }

    # Verify the top of the inbox (real vs hallucinated).
    top = opportunities[:12]
    verdicts = await verify_urls([o["source_url"] for o in top])
    for o in top:
        v = verdicts.get(o["source_url"], {})
        o["url_verdict"] = v.get("verdict", "unverifiable")
        o["final_url"] = v.get("final_url")
    audit["opportunities"] = opportunities

    platform_mix: dict[str, int] = {}
    for o in opportunities:
        platform_mix[o["platform"]] = platform_mix.get(o["platform"], 0) + 1
    verdict_counts: dict[str, int] = {}
    for o in top:
        verdict_counts[o["url_verdict"]] = verdict_counts.get(o["url_verdict"], 0) + 1

    return {
        "audit_id": audit_id,
        "total_opportunities": len(opportunities),
        "platform_mix": platform_mix,
        "url_verification": verdict_counts,
        "top_opportunities": [
            {
                "url": o["source_url"],
                "platform": o["platform"],
                "title": o["title"],
                "engines": o["engines"],
                "citations": o["citation_count"],
                "prompts_citing_it": o["prompt_breadth"],
                "score": o["score"],
                "url_verdict": o["url_verdict"],
            }
            for o in top
        ],
        "next_step": "Call draft_growth_actions to generate platform-tailored content for the top real sources.",
    }


# ==============================================================================
# TOOL 5 — Platform-branched growth drafting
# ==============================================================================
async def draft_growth_actions(audit_id: str, top_n: int = 5) -> dict:
    """Generate platform-tailored growth content for the top verified sources.

    Branches by platform: a Reddit/forum citation gets an authentic reply
    draft; a Q&A page gets an expert answer; a YouTube video gets a comment +
    video brief; a review platform gets an acquisition play; a blog gets a
    counter-content brief + outreach pitch. Hallucinated URLs are skipped.

    Args:
        audit_id: The audit session id returned by generate_audit_prompts.
        top_n: How many top opportunities to draft for (default 5).

    Returns:
        dict with ready-to-use drafts grouped by platform play.
    """
    audit = _AUDITS.get(audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id}'. Call generate_audit_prompts first."}
    opportunities = audit.get("opportunities") or []
    if not opportunities:
        return {"error": "No opportunities mined yet. Call find_growth_opportunities first."}

    usable = [o for o in opportunities if o.get("url_verdict") != "hallucinated"][: max(1, min(top_n, 8))]
    drafts = []
    for o in usable:
        d = await draft_growth_action(o, audit["brand_name"], audit["domain"], audit["topic"])
        drafts.append({
            "url": o["source_url"],
            "platform": o["platform"],
            "url_verdict": o.get("url_verdict", "unverified"),
            "score": o["score"],
            **d,
        })
    audit["growth_drafts"] = drafts

    by_play: dict[str, int] = {}
    for d in drafts:
        by_play[d["action_type"]] = by_play.get(d["action_type"], 0) + 1

    return {
        "audit_id": audit_id,
        "drafts_generated": len(drafts),
        "plays": by_play,
        "drafts": drafts,
    }


# ==============================================================================
# ROOT AGENT
# ==============================================================================
root_agent = Agent(
    name="sanbi_audit_agent",
    model="gemini-2.5-flash",
    description=(
        "Sanbi AI visibility auditor: measures whether AI assistants "
        "(ChatGPT, Gemini) recommend a brand or its competitors, and why."
    ),
    instruction="""You are Sanbi, an AI visibility auditor.

When a user asks about a brand's AI visibility (e.g. "How visible is
sight360.com for LASIK surgery?"), run the full audit pipeline:

1. Call generate_audit_prompts(brand_domain, topic) — this researches the
   company and plans realistic branded + unbranded audit queries.
   Briefly tell the user what you learned about the brand and which
   queries you'll audit.

2. Call query_engines(audit_id) — this asks every query across OpenAI and
   Vertex Gemini independently. Summarize what the engines said at a glance.

3. Call grade_responses(audit_id) — this grades every response and builds
   the competitive leaderboard.

After presenting the audit results, ACT on them:

4. Call find_growth_opportunities(audit_id) — this classifies every cited
   source by platform (reddit, forum, Q&A, youtube, reviews, blog, wiki...),
   ranks them by Sanbi's replyability-weighted score, and verifies the top
   URLs are real (AI engines sometimes hallucinate citations — call out any
   hallucinated ones explicitly).

5. Call draft_growth_actions(audit_id) — this generates a different kind of
   growth content for each surface: authentic reply drafts for forums and
   reddit, expert answers for Q&A, comment + video briefs for youtube, review
   acquisition plays for review platforms, counter-content briefs for blogs.

Then present the full picture:
- Overall visibility score and rate for the brand
- Top competitors stealing share of voice (from the leaderboard)
- Per-engine differences (e.g. visible on Gemini but invisible on ChatGPT)
- The biggest visibility gaps (unbranded queries where the brand never appears)
- The growth inbox: where AI actually cites from (platform mix), which
  citations were hallucinated, and the drafted plays grouped by type
  (community replies, expert answers, counter-content, review plays)

Be concise and quantitative. Always run steps 1-3 in order — never skip
grading. Run steps 4-5 by default unless the user only wants the audit. If
the user gives just a domain with no topic, infer a sensible topic from the
brand's industry after step 1.""",
    tools=[
        generate_audit_prompts,
        query_engines,
        grade_responses,
        find_growth_opportunities,
        draft_growth_actions,
    ],
)
