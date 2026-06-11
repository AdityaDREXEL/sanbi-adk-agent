"""
agents/sanbi_audit/agent.py — Sanbi multi-agent system (ADK).

Agent tree:

    sanbi_coordinator (root, no tools — routes)
    ├── audit_agent   MEASURE: research → multi-engine queries → grading
    │                 tools: generate_audit_prompts, query_engines, grade_responses
    └── growth_agent  ACT: classify → rank → verify citations → draft content
                      tools: find_growth_opportunities, draft_growth_actions

State: tools share audit data through ADK **session state** (ToolContext.state)
under "audit:<id>" keys — never through the model's context window. Raw engine
responses run 5-15k chars each; the model only ever sees compact summaries.
Both sub-agents read the same session state, which is what makes the
audit → growth handoff work. State lives in whatever SessionService the runner
provides (in-memory in the dev UI; swap in a persistent service for production
without touching tool code).

"active_audit_id" tracks the most recent audit so follow-up tools work even
when the model omits the id.

Run locally:
    adk web agents          # from repo root → http://localhost:8000
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

# Ensure the repo root (which holds the `sanbi_core` package) is importable.
# `adk web agents` only adds the agents/ directory to sys.path, so without this
# bootstrap the repo-root `sanbi_core` package is invisible when ADK imports
# this module. Same story in the Cloud Run image (WORKDIR /app). parents[2]:
#   .../agents/sanbi_audit/agent.py -> .../agents/sanbi_audit -> .../agents -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from google.adk.agents import Agent
from google.adk.tools import ToolContext

from sanbi_core.planning import analyze_brand_identity, auto_generate_brand_prompts
from sanbi_core.execution import query_all_engines, ENGINES
from sanbi_core.analysis import grade_result, build_leaderboard, generate_executive_summary
from sanbi_core.growth import build_opportunities, draft_growth_action
from sanbi_core.verifier import verify_urls

logger = logging.getLogger(__name__)

_ACTIVE_KEY = "active_audit_id"


def _audit_key(audit_id: str) -> str:
    return f"audit:{audit_id}"


def _resolve_audit(tool_context: ToolContext, audit_id: str = "") -> tuple[str, Optional[dict]]:
    """Resolve an audit by id, falling back to the session's active audit."""
    if tool_context is None:  # defensive: ADK always injects in practice
        return "", None
    aid = (audit_id or "").strip() or tool_context.state.get(_ACTIVE_KEY, "")
    if not aid:
        return "", None
    return aid, tool_context.state.get(_audit_key(aid))


def _save_audit(tool_context: ToolContext, audit_id: str, audit: dict) -> None:
    """Write the audit to session state (assignment is what State delta-tracks)."""
    tool_context.state[_audit_key(audit_id)] = audit
    tool_context.state[_ACTIVE_KEY] = audit_id


# ==============================================================================
# TOOL 1 — Plan the audit
# ==============================================================================
async def generate_audit_prompts(
    brand_domain: str,
    topic: str,
    location: Optional[str] = None,
    tool_context: ToolContext = None,
) -> dict:
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
    domain_clean = (brand_domain or "").strip().rstrip("/")
    bare = domain_clean.replace("https://", "").replace("http://", "")
    if not bare or "." not in bare or " " in bare:
        return {
            "error": f"'{brand_domain}' doesn't look like a domain. "
                     "Ask the user for a website domain like 'sight360.com'."
        }
    if tool_context is None:
        return {"error": "Internal: tool context missing. Retry the request."}

    loc = location or "United States"
    identity = await analyze_brand_identity(domain_clean)
    prompts = await auto_generate_brand_prompts(
        user_topic=topic,
        domain=domain_clean,
        location=loc,
        max_count=6,
        branded_count=2,
        unbranded_count=4,
        identity=identity,
    )

    audit_id = uuid.uuid4().hex[:12]
    audit = {
        "domain": domain_clean,
        "brand_name": identity.get("brand_name", domain_clean.split(".")[0].title()),
        "topic": topic,
        "location": loc,
        "identity": identity,
        "prompts": prompts,
        "responses": [],   # filled by query_engines
        "audit_log": [],   # filled by grade_responses
    }
    _save_audit(tool_context, audit_id, audit)

    return {
        "audit_id": audit_id,
        "brand_name": audit["brand_name"],
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
async def query_engines(audit_id: str = "", tool_context: ToolContext = None) -> dict:
    """Run every planned prompt across all AI engines (OpenAI + Vertex Gemini).

    Each engine answers independently using its native capabilities (Gemini
    uses live Google Search grounding; OpenAI answers from training data).
    Raw responses are stored in session state; this returns compact previews.

    Args:
        audit_id: The audit session id. Empty = the most recent audit in
            this session.

    Returns:
        dict with per-prompt, per-engine response previews and citation counts.
    """
    aid, audit = _resolve_audit(tool_context, audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id or aid}'. Call generate_audit_prompts first."}

    # All prompts in parallel; each query_all_engines already fans out across
    # engines internally and never raises (per-engine failures are isolated).
    prompts = audit["prompts"]
    all_engine_results = await asyncio.gather(
        *[query_all_engines(p["search_query"], audit["location"]) for p in prompts]
    )

    responses = []
    previews = []
    for p, engine_results in zip(prompts, all_engine_results):
        prompt_text = p["search_query"]
        for engine, res in engine_results.items():
            responses.append({
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

    audit["responses"] = responses
    _save_audit(tool_context, aid, audit)

    return {
        "audit_id": aid,
        "engines": ENGINES,
        "responses_collected": len(responses),
        "previews": previews,
        "next_step": "Call grade_responses with this audit_id.",
    }


# ==============================================================================
# TOOL 3 — Grade + leaderboard
# ==============================================================================
async def grade_responses(audit_id: str = "", tool_context: ToolContext = None) -> dict:
    """Grade every engine response and build the competitive leaderboard.

    For each stored response, an LLM grader determines: is the brand visible,
    at what rank, with what sentiment, and which competitors appear. Scores
    use Sanbi's weighted formula (rank decay x sentiment multiplier). Returns
    the leaderboard, per-engine breakdown, and executive summary.

    Args:
        audit_id: The audit session id. Empty = the most recent audit in
            this session.

    Returns:
        dict with brand score, competitor leaderboard, per-engine stats,
        gap analysis (prompts where the brand was invisible), and summary.
    """
    aid, audit = _resolve_audit(tool_context, audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id or aid}'. Call generate_audit_prompts first."}
    if not audit["responses"]:
        return {"error": "No responses collected yet. Call query_engines first."}

    # Grade all responses in parallel (grade_result never raises — it returns
    # _empty_grade() on failure). Semaphore caps concurrent grader calls.
    sem = asyncio.Semaphore(8)

    async def _graded(entry):
        async with sem:
            return await grade_result(entry["prompt"], entry["response"], audit["domain"])

    grades = await asyncio.gather(*[_graded(e) for e in audit["responses"]])

    audit_log = [
        {
            "prompt": entry["prompt"],
            "prompt_type": entry["prompt_type"],
            "engine": entry["engine"],
            "grade": grade,
        }
        for entry, grade in zip(audit["responses"], grades)
    ]
    audit["audit_log"] = audit_log
    _save_audit(tool_context, aid, audit)

    leaderboard = build_leaderboard(audit_log, audit["brand_name"])
    summary = await generate_executive_summary(audit["domain"], audit_log)

    gaps = [
        {"prompt": e["prompt"], "engine": e["engine"], "type": e["prompt_type"]}
        for e in audit_log
        if not e["grade"].get("is_visible")
    ]

    return {
        "audit_id": aid,
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
async def find_growth_opportunities(audit_id: str = "", tool_context: ToolContext = None) -> dict:
    """Find, rank, and verify the community sources AI engines cited.

    Runs every citation from the graded audit through Sanbi's deterministic
    platform classifier (reddit/forum/qa/youtube/reviews/blog/wiki/...),
    aggregates citations per URL, and ranks them with the production scoring
    formula — (engines x 25 + prompt_breadth x 5 + recency x 20) x platform
    replyability weight. Then verifies the top URLs are REAL (HEAD checks,
    YouTube oEmbed) — AI engines sometimes hallucinate citations.

    Args:
        audit_id: The audit session id. Empty = the most recent audit in
            this session.

    Returns:
        dict with platform mix, hallucination check results, and the ranked
        top opportunities (each with url, platform, score, url_verdict).
    """
    aid, audit = _resolve_audit(tool_context, audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id or aid}'. Call generate_audit_prompts first."}
    if not audit["audit_log"]:
        return {"error": "Audit not graded yet. Call grade_responses first."}

    opportunities = build_opportunities(audit["audit_log"])
    if not opportunities:
        return {
            "audit_id": aid,
            "total_opportunities": 0,
            "note": "No community/growth surfaces were cited in this audit.",
        }

    # Verify the top of the inbox (real vs hallucinated). A total verification
    # failure (network down) must degrade, not kill the audit.
    top = opportunities[:12]
    try:
        verdicts = await verify_urls([o["source_url"] for o in top])
    except Exception as e:
        logger.error(f"URL verification failed wholesale: {e}")
        verdicts = {}
    for o in top:
        v = verdicts.get(o["source_url"], {})
        o["url_verdict"] = v.get("verdict", "unverifiable")
        o["final_url"] = v.get("final_url")
    audit["opportunities"] = opportunities
    _save_audit(tool_context, aid, audit)

    platform_mix: dict[str, int] = {}
    for o in opportunities:
        platform_mix[o["platform"]] = platform_mix.get(o["platform"], 0) + 1
    verdict_counts: dict[str, int] = {}
    for o in top:
        verdict_counts[o["url_verdict"]] = verdict_counts.get(o["url_verdict"], 0) + 1

    return {
        "audit_id": aid,
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
async def draft_growth_actions(
    audit_id: str = "",
    top_n: int = 5,
    tool_context: ToolContext = None,
) -> dict:
    """Generate platform-tailored growth content for the top verified sources.

    Branches by platform: a Reddit/forum citation gets an authentic reply
    draft; a Q&A page gets an expert answer; a YouTube video gets a comment +
    video brief; a review platform gets an acquisition play; a blog gets a
    counter-content brief + outreach pitch. Hallucinated URLs are skipped.

    Args:
        audit_id: The audit session id. Empty = the most recent audit in
            this session.
        top_n: How many top opportunities to draft for (default 5).

    Returns:
        dict with ready-to-use drafts grouped by platform play.
    """
    aid, audit = _resolve_audit(tool_context, audit_id)
    if not audit:
        return {"error": f"Unknown audit_id '{audit_id or aid}'. Call generate_audit_prompts first."}
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
    _save_audit(tool_context, aid, audit)

    by_play: dict[str, int] = {}
    for d in drafts:
        by_play[d["action_type"]] = by_play.get(d["action_type"], 0) + 1

    return {
        "audit_id": aid,
        "drafts_generated": len(drafts),
        "plays": by_play,
        "drafts": drafts,
    }


# ==============================================================================
# AGENT TREE
# ==============================================================================
_MODEL = "gemini-2.5-flash"

audit_agent = Agent(
    name="audit_agent",
    model=_MODEL,
    description=(
        "Measurement specialist: researches a brand, audits it across OpenAI "
        "and Vertex Gemini, grades every response, and builds the competitive "
        "visibility leaderboard."
    ),
    instruction="""You are Sanbi's audit specialist. You MEASURE a brand's AI visibility.

Run the measurement pipeline in strict order:

1. generate_audit_prompts(brand_domain, topic) — researches the company and
   plans realistic branded + unbranded audit queries. Briefly tell the user
   what you learned about the brand and which queries you'll audit.

2. query_engines() — asks every query across OpenAI and Vertex Gemini
   independently. Summarize what the engines said at a glance.

3. grade_responses() — grades every response and builds the leaderboard.

Then present, concisely and quantitatively:
- Overall visibility score and rate for the brand
- Top competitors stealing share of voice
- Per-engine differences (e.g. visible on Gemini but invisible on ChatGPT)
- The biggest gaps (unbranded queries where the brand never appears)

Never skip grading. If the user gives just a domain with no topic, infer a
sensible topic from the brand's industry after step 1.

When the audit is done, offer the growth phase. If the user wants growth
opportunities, citation analysis, or content drafts, transfer to growth_agent
— the audit data is already in this session's state.""",
    tools=[generate_audit_prompts, query_engines, grade_responses],
)

growth_agent = Agent(
    name="growth_agent",
    model=_MODEL,
    description=(
        "Action specialist: classifies the sources AI engines cited, ranks "
        "them by replyability, verifies they are real (anti-hallucination), "
        "and drafts platform-tailored growth actions."
    ),
    instruction="""You are Sanbi's growth specialist. You ACT on a completed audit.

You need a graded audit in this session. If the tools report there is none,
transfer to audit_agent to run the measurement first.

1. find_growth_opportunities() — classifies every cited source by platform
   (reddit, forum, Q&A, youtube, reviews, blog, wiki...), ranks them by
   Sanbi's replyability-weighted score, and verifies the top URLs are real.
   AI engines sometimes hallucinate citations — call out any hallucinated
   URLs explicitly.

2. draft_growth_actions() — generates a different kind of growth content for
   each surface: authentic reply drafts for forums and reddit, expert answers
   for Q&A, comment + video briefs for youtube, review acquisition plays for
   review platforms, counter-content briefs for blogs.

Present the growth inbox clearly: platform mix, verification verdicts (flag
hallucinations prominently), then the drafted plays grouped by type
(community replies, expert answers, counter-content, review plays).""",
    tools=[find_growth_opportunities, draft_growth_actions],
)

root_agent = Agent(
    name="sanbi_coordinator",
    model=_MODEL,
    description=(
        "Sanbi AI-visibility coordinator: measures whether AI assistants "
        "recommend a brand, then turns the findings into growth actions. "
        "Routes between the audit (measure) and growth (act) specialists."
    ),
    instruction="""You are Sanbi, an AI-visibility coordinator. Sanbi measures whether AI
assistants (ChatGPT, Gemini) recommend a brand — and then turns those findings
into concrete growth actions.

You do not run tools yourself. You route:

- Audit / visibility requests ("audit X", "how visible is X for Y?") →
  transfer to audit_agent.
- Growth requests on a completed audit ("where do AI engines cite from?",
  "verify those sources", "draft growth actions / replies") →
  transfer to growth_agent.

For a combined request like "audit X and draft growth actions for the top
sources", start with audit_agent — the specialists hand off between
themselves and share this session's audit state.

If the user asks what you can do, explain the measure → act pipeline in two
sentences, then ask for a brand domain and topic.""",
    sub_agents=[audit_agent, growth_agent],
)
