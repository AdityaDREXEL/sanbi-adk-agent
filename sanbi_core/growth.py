"""
sanbi_core/growth.py — Growth opportunity mining + platform-branched drafting.

Replicates production Sanbi's Growth feature (growth/materializer.py + the v2
drafting design) in-memory, operating on a single audit run instead of a
database of historical logs.

Pipeline:
  1. CLASSIFY — every citation URL through the deterministic tiered classifier
     (sanbi_core.platforms — same code as production). No LLM call needed.
  2. SCORE — aggregate citations per URL and rank with the production formula:
         (effective_engines×25 + prompt_breadth×5 + recency×20) × platform_weight
     Platform weights encode "replyability": reddit 1.30 … blog 0.50 … wiki 0.35.
  3. VERIFY — (sanbi_core.verifier) prove each top URL is real, not hallucinated.
  4. DRAFT — platform-branched growth content. A Reddit citation gets an
     authentic reply draft; a blog citation gets a counter-content brief; a
     review-site citation gets an acquisition playbook. Different surfaces
     need different motions — you can't blog your way into a forum thread.
"""

import logging
from typing import Any, Dict, List

from .platforms import classify_url, platform_weight
from .gemini import robust_json_call

logger = logging.getLogger(__name__)


# ==============================================================================
# 1+2. Classify + score  (in-memory materializer)
# ==============================================================================
def build_opportunities(audit_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Bucket every cited URL from a graded audit into ranked growth opportunities.

    Production formula, preserved exactly:
        raw   = effective_engines*25 + prompt_breadth*5 + recency*20
        score = raw * platform_weight(platform)

    Notes:
      - A URL cited by ONLY Perplexity counts as 0.5 engines in production
        (Perplexity cites broadly and cheaply). Kept for parity even though
        this build runs OpenAI + Gemini only.
      - In a live audit every citation is from "now", so recency contributes
        its full 20 points. The production system decays it linearly over 30d.
    """
    buckets: Dict[str, Dict[str, Any]] = {}

    for entry in audit_log:
        grade = entry.get("grade", {})
        engine = (entry.get("engine") or "").lower()
        prompt = entry.get("prompt") or ""
        titles = grade.get("source_titles", {}) or {}

        for url in grade.get("cited_sources", []) or []:
            url = (url or "").strip()
            if not url:
                continue
            domain, platform = classify_url(url)
            if not domain or not platform:
                continue  # not a community/growth surface (or blocked)

            b = buckets.get(url)
            if b is None:
                b = {
                    "source_url": url,
                    "source_domain": domain,
                    "platform": platform,
                    "title": None,
                    "engines": set(),
                    "prompts": set(),
                    "citation_count": 0,
                }
                buckets[url] = b
            if not b["title"]:
                t = (titles.get(url) or "").strip()
                if t and t.lower() not in ("source", "mentioned in text", "untitled"):
                    b["title"] = t
            if engine:
                b["engines"].add(engine)
            if prompt:
                b["prompts"].add(prompt)
            b["citation_count"] += 1

    opportunities = []
    for url, b in buckets.items():
        engine_breadth = len(b["engines"])
        only_perplexity = engine_breadth == 1 and "perplexity" in b["engines"]
        effective_engines = 0.5 if only_perplexity else float(engine_breadth)
        prompt_breadth = len(b["prompts"])
        recency = 1.0  # live audit — all citations are from right now

        raw = (effective_engines * 25.0) + (prompt_breadth * 5.0) + (recency * 20.0)
        score = round(raw * platform_weight(b["platform"]), 2)

        opportunities.append({
            "source_url": url,
            "source_domain": b["source_domain"],
            "platform": b["platform"],
            "title": b["title"],
            "engines": sorted(b["engines"]),
            "prompts": sorted(b["prompts"]),
            "citation_count": b["citation_count"],
            "prompt_breadth": prompt_breadth,
            "score": score,
        })

    opportunities.sort(key=lambda o: o["score"], reverse=True)
    return opportunities


# ==============================================================================
# 4. Platform-branched drafting  (production Growth v2 design)
# ==============================================================================
# Each playbook tells the drafting LLM what motion fits the surface. Grouped
# by *replyability*, mirroring the platform-weight tiers.
_PLAYBOOKS: Dict[str, Dict[str, str]] = {
    "community_reply": {
        "platforms": "reddit hn forum",
        "action_type": "community_reply",
        "instruction": (
            "Draft an authentic, genuinely helpful reply for this discussion "
            "thread. Lead with real substance that answers the question — "
            "specifics, trade-offs, first-hand-style detail. Mention the brand "
            "at most once, naturally, where a knowledgeable practitioner would. "
            "Match the platform's native tone (no marketing voice, no emoji "
            "spam). The goal: a reply so useful the community upvotes it and "
            "AI engines cite it."
        ),
    },
    "expert_answer": {
        "platforms": "qa quora stackexchange",
        "action_type": "expert_answer",
        "instruction": (
            "Draft an expert answer for this Q&A page: direct answer first, "
            "then reasoning, then caveats. Cite verifiable facts. Brand "
            "mention only if it's the honest answer to the question; disclose "
            "affiliation in one short line, as these platforms require."
        ),
    },
    "video_engagement": {
        "platforms": "youtube",
        "action_type": "video_engagement",
        "instruction": (
            "Two deliverables: (1) a top-comment draft for this video that adds "
            "real value (a correction, an addition, a practical tip) so it "
            "earns visibility, and (2) a 5-bullet brief for a response/companion "
            "video the brand could publish to capture the same AI citations."
        ),
    },
    "review_acquisition": {
        "platforms": "reviews",
        "action_type": "review_acquisition",
        "instruction": (
            "AI engines cite this review platform for buying decisions. Produce "
            "a review acquisition play: why this platform matters for the "
            "brand, who to ask for reviews and when (timing in the customer "
            "journey), and a short, compliant review-request message template. "
            "Never suggest fake or incentivized-without-disclosure reviews."
        ),
    },
    "counter_content": {
        "platforms": "blog medium tutorial devto",
        "action_type": "counter_content",
        "instruction": (
            "This editorial page is being cited by AI engines but likely "
            "doesn't accept comments. Produce: (1) an article brief for the "
            "brand's own site engineered to win the same citations — H1, the "
            "exact questions it must answer, key entities to include, schema "
            "markup recommendation; and (2) a 3-sentence guest-post/inclusion "
            "pitch to the publication that owns this page."
        ),
    },
    "social_engagement": {
        "platforms": "x linkedin instagram tiktok facebook producthunt github",
        "action_type": "social_engagement",
        "instruction": (
            "Draft a substantive comment/reply for this social post from the "
            "brand's perspective, plus one original post idea on the same "
            "topic that could attract the same AI citations."
        ),
    },
    "reference_note": {
        "platforms": "wiki",
        "action_type": "reference_note",
        "instruction": (
            "This is reference material (wiki-style) — you can't reply. "
            "Identify what claim it likely supports, and outline what citable, "
            "neutral source material the brand should publish (data, studies, "
            "documentation) that could legitimately be referenced there."
        ),
    },
}

_PLATFORM_TO_PLAYBOOK: Dict[str, Dict[str, str]] = {
    p: pb for pb in _PLAYBOOKS.values() for p in pb["platforms"].split()
}


def get_playbook(platform: str) -> Dict[str, str]:
    return _PLATFORM_TO_PLAYBOOK.get(platform, _PLAYBOOKS["community_reply"])


async def draft_growth_action(
    opportunity: Dict[str, Any],
    brand_name: str,
    domain: str,
    topic: str,
) -> Dict[str, Any]:
    """Generate the platform-appropriate growth action for one opportunity."""
    playbook = get_playbook(opportunity["platform"])
    prompts_ctx = opportunity.get("prompts", [])[:3]

    reasoning_prompt = f"""
    You are Sanbi's growth strategist for {brand_name} ({domain}). Topic: {topic}.

    AI engines ({", ".join(opportunity.get("engines", []))}) cited this source when
    answering real buyer questions:
      URL: {opportunity["source_url"]}
      Platform type: {opportunity["platform"]}
      Page title: {opportunity.get("title") or "unknown"}
      Cited when users asked: {prompts_ctx}

    YOUR TASK ({playbook["action_type"]}):
    {playbook["instruction"]}

    Hard rules: be useful first, promotional last. Never astroturf — write as
    someone who would disclose their affiliation. Keep drafts ready to paste.
    """

    json_instruction = """
    Format into this strict JSON:
    {
      "action_type": "string",
      "headline": "one-line description of the play",
      "draft": "the ready-to-use content (reply / answer / brief / playbook)",
      "why_this_source": "1-2 sentences: why acting here moves AI visibility",
      "effort": "low | medium | high"
    }
    """

    try:
        data = await robust_json_call(reasoning_prompt, json_instruction, use_search=False)
        data["action_type"] = data.get("action_type") or playbook["action_type"]
        return data
    except Exception as e:
        logger.error(f"Growth draft failed for {opportunity['source_url']}: {e}")
        return {
            "action_type": playbook["action_type"],
            "headline": "Draft generation failed",
            "draft": "",
            "why_this_source": "",
            "effort": "unknown",
        }
