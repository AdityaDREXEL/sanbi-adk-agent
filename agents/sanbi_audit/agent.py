"""
agents/sanbi_audit/agent.py — SanbiAuditAgent (ADK root agent).

A single ADK agent with three tools that walk the full Sanbi brand-visibility
audit pipeline:

    1. generate_audit_prompts  → researches the brand, plans audit queries
    2. query_engines           → runs every prompt across OpenAI + Vertex Gemini
    3. grade_responses         → LLM-grades each response, builds leaderboard

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

Then present the results clearly:
- Overall visibility score and rate for the brand
- Top competitors stealing share of voice (from the leaderboard)
- Per-engine differences (e.g. visible on Gemini but invisible on ChatGPT)
- The biggest visibility gaps (unbranded queries where the brand never appears)
- 2-3 concrete recommendations from the executive summary

Be concise and quantitative. Always run all three steps in order — never skip
grading. If the user gives just a domain with no topic, infer a sensible topic
from the brand's industry after step 1.""",
    tools=[generate_audit_prompts, query_engines, grade_responses],
)
