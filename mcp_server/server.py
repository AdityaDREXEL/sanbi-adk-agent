"""
mcp_server/server.py — Sanbi MCP server.

Exposes Sanbi's brand-visibility audit as a Model Context Protocol tool so ANY
MCP-capable agent (Claude Desktop, Gemini CLI, custom ADK agents, IDEs) can
run an audit. This is the "agent-as-infrastructure" pattern: the same
sanbi_core business logic that powers the ADK agent, exposed over an open
protocol.

Tool:
    run_visibility_audit(domain, topic, location) → full audit JSON
        (identity → prompts → multi-engine query → grading → leaderboard)

Run (stdio, for Claude Desktop / MCP Inspector):
    python -m mcp_server.server

Run (HTTP, for Cloud Run):
    MCP_TRANSPORT=http MCP_PORT=8081 python -m mcp_server.server
"""

import logging
import os

from mcp.server.fastmcp import FastMCP

from sanbi_core.planning import analyze_brand_identity, auto_generate_brand_prompts
from sanbi_core.execution import query_all_engines
from sanbi_core.analysis import grade_result, build_leaderboard, generate_executive_summary
from sanbi_core.growth import build_opportunities
from sanbi_core.verifier import verify_urls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "sanbi-visibility-audit",
    instructions=(
        "Sanbi measures brand visibility inside AI assistants. "
        "Call run_visibility_audit with a brand domain and topic to get a "
        "competitive leaderboard of who AI engines actually recommend."
    ),
)


@mcp.tool()
async def run_visibility_audit(
    domain: str,
    topic: str = "",
    location: str = "United States",
) -> dict:
    """Run a full AI-visibility audit for a brand.

    Researches the brand, generates realistic branded + unbranded audit
    queries, asks them across multiple AI engines (OpenAI + Vertex Gemini),
    grades every response for brand visibility/rank/sentiment, and returns a
    competitive leaderboard plus a ranked, URL-verified growth inbox of the
    community sources AI engines actually cited.

    Args:
        domain: Brand website domain, e.g. "sight360.com".
        topic: Audit focus, e.g. "LASIK surgery". Empty = inferred from research.
        location: Market to simulate, e.g. "United States".

    Returns:
        Full audit JSON: brand identity, prompts, leaderboard, per-engine
        breakdown, visibility gaps, executive summary.
    """
    logger.info(f"MCP audit start: domain={domain} topic={topic!r} location={location}")

    # 1. Plan
    identity = await analyze_brand_identity(domain)
    brand_name = identity.get("brand_name", domain.split(".")[0].title())
    effective_topic = topic or identity.get("industry") or "general"
    prompts = await auto_generate_brand_prompts(
        user_topic=effective_topic,
        domain=domain,
        location=location,
        max_count=6,
        branded_count=2,
        unbranded_count=4,
        identity=identity,
    )

    # 2. Execute (every prompt × every engine)
    audit_log = []
    for p in prompts:
        engine_results = await query_all_engines(p["search_query"], location)
        for engine, res in engine_results.items():
            grade = await grade_result(p["search_query"], res, domain)
            audit_log.append({
                "prompt": p["search_query"],
                "prompt_type": p["prompt_type"],
                "engine": engine,
                "grade": grade,
            })

    # 3. Aggregate
    leaderboard = build_leaderboard(audit_log, brand_name)
    summary = await generate_executive_summary(domain, audit_log)
    gaps = [
        {"prompt": e["prompt"], "engine": e["engine"]}
        for e in audit_log if not e["grade"].get("is_visible")
    ]

    # 4. Growth inbox: classify + rank cited sources, verify the top URLs
    opportunities = build_opportunities(audit_log)
    top_opps = opportunities[:10]
    if top_opps:
        verdicts = await verify_urls([o["source_url"] for o in top_opps])
        for o in top_opps:
            o["url_verdict"] = verdicts.get(o["source_url"], {}).get("verdict", "unverifiable")
    platform_mix: dict = {}
    for o in opportunities:
        platform_mix[o["platform"]] = platform_mix.get(o["platform"], 0) + 1

    logger.info(
        f"MCP audit done: {brand_name} score={leaderboard['brand']['avg_visibility_score']} "
        f"rate={leaderboard['brand']['visibility_rate']}"
    )

    return {
        "brand": brand_name,
        "domain": domain,
        "topic": effective_topic,
        "identity": {
            "industry": identity.get("industry"),
            "target_audience": identity.get("target_audience"),
            "competitors": identity.get("competitors", [])[:5],
        },
        "prompts_audited": [p["search_query"] for p in prompts],
        "leaderboard": leaderboard,
        "visibility_gaps": gaps[:10],
        "growth_inbox": {
            "total_opportunities": len(opportunities),
            "platform_mix": platform_mix,
            "top_opportunities": [
                {
                    "url": o["source_url"],
                    "platform": o["platform"],
                    "engines": o["engines"],
                    "citations": o["citation_count"],
                    "score": o["score"],
                    "url_verdict": o.get("url_verdict", "unverified"),
                }
                for o in top_opps
            ],
        },
        "executive_summary": summary,
    }


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport in ("http", "streamable-http"):
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.getenv("MCP_PORT", "8081"))
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio — Claude Desktop / MCP Inspector
