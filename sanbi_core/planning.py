"""
sanbi_core/planning.py — Audit planning: brand identity research + prompt generation.

Adapted from production Sanbi `core/planning.py`.

Pipeline:
  1. RESEARCH  — Google-Search-grounded deep-dive into the company (Vertex Gemini)
  2. EXTRACT   — Parse raw research into structured identity JSON
  3. VALIDATE  — Catch generic / wrong classifications
  4. GENERATE  — Produce branded + unbranded audit queries real users would type
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List

from .gemini import gemini_grounded_text, gemini_force_json, robust_json_call

logger = logging.getLogger(__name__)


# ==============================================================================
# STEP 1: MULTI-STEP BRAND IDENTITY PIPELINE
# ==============================================================================

async def analyze_brand_identity(domain: str) -> Dict[str, Any]:
    """
    Multi-step brand identity pipeline. Returns a rich identity dict with:
    brand_name, industry, specialty, business_model, target_audience,
    audience_pain_points, competitors, product_categories, key_search_terms.
    """
    logger.info(f"🔍 [Brand Identity] Starting multi-step analysis for: {domain}")

    clean_domain = domain.replace("https://", "").replace("http://", "").replace("www.", "")
    brand_guess = clean_domain.split('.')[0]

    # ── STEP 1: Deep Research via Google Search ──────────────────────────
    research_prompt = f"""
    Research this company thoroughly: "{clean_domain}"

    Search for:
    1. "{clean_domain}" — what does their official website say they do?
    2. "{brand_guess} company what do they do" — understand their core business
    3. "{brand_guess} reviews" — what do real customers say about them?

    REPORT THESE FACTS (be extremely specific, never generic):

    A) EXACT BUSINESS: What does this company ACTUALLY sell or offer?
    B) SPECIFIC INDUSTRY NICHE: What exact sub-industry are they in?
       - Not "health" but "health insurance marketplace" or "medical device manufacturer"
    C) TARGET CUSTOMERS: Who specifically buys from them?
    D) DIRECT COMPETITORS: Name 3-5 real companies competing in the EXACT same niche
    E) PRODUCTS/SERVICES: List the specific things they sell or offer
    F) CUSTOMER PROBLEMS: What pain points do their customers have that this company solves?
    """

    raw_research, _sources = await gemini_grounded_text(research_prompt, use_search=True)

    if not raw_research or len(raw_research) < 50:
        logger.warning(f"⚠️ Research returned insufficient data for {domain}")
        return _fallback_identity(domain)

    # ── STEP 2: Structured Extraction ────────────────────────────────────
    extraction_context = f"""
    Based on this research about "{clean_domain}", extract a precise brand identity.

    RESEARCH DATA:
    {raw_research[:20000]}
    """

    json_instruction = """
    Extract into strict JSON. Be HYPER-SPECIFIC — never use generic filler terms:
    {
        "brand_name": "Official company name with proper spacing and capitalization",
        "industry": "SPECIFIC industry niche (e.g. 'Health Insurance Marketplace' NOT 'Health')",
        "specialty": "Their unique value proposition",
        "business_model": "Exactly how they make money",
        "target_audience": "Specific buyer persona",
        "audience_pain_points": ["Specific problem 1", "Specific problem 2", "Specific problem 3"],
        "competitors": ["Real Competitor 1", "Real Competitor 2", "Real Competitor 3"],
        "product_categories": ["Specific product/service 1", "Specific product/service 2"],
        "key_search_terms": ["Real search query 1", "Real search query 2", "Real search query 3"]
    }
    """

    try:
        identity = await gemini_force_json(extraction_context, json_instruction)
        if identity and identity.get("brand_name"):
            identity = _validate_identity(identity, domain)
            logger.info(
                f"✅ Identity resolved: {identity.get('brand_name')} | "
                f"Industry: {identity.get('industry')} | "
                f"Audience: {identity.get('target_audience')}"
            )
            return identity
    except Exception as e:
        logger.error(f"Identity extraction failed: {e}")

    return _fallback_identity(domain)


def _validate_identity(identity: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """Validate and fix common AI mistakes in identity extraction."""
    defaults = {
        "brand_name": domain.replace("https://", "").replace("http://", "").replace("www.", "").split('.')[0].capitalize(),
        "industry": "",
        "specialty": "",
        "business_model": "",
        "target_audience": "",
        "audience_pain_points": [],
        "competitors": [],
        "product_categories": [],
        "key_search_terms": [],
    }
    for key, default in defaults.items():
        if key not in identity or not identity[key]:
            identity[key] = default

    generic_industries = {
        "technology", "business", "services", "company", "online", "digital",
        "web", "internet", "general", "commerce", "e-commerce",
    }
    if identity.get("industry", "").lower().strip() in generic_industries:
        logger.warning(f"⚠️ Industry too generic: '{identity['industry']}'")

    return identity


def _fallback_identity(domain: str) -> Dict[str, Any]:
    """Last-resort identity when the entire pipeline fails."""
    name = domain.replace("https://", "").replace("http://", "").replace("www.", "").split('.')[0].capitalize()
    return {
        "brand_name": name,
        "industry": "",
        "specialty": "",
        "business_model": "",
        "target_audience": "",
        "audience_pain_points": [],
        "competitors": [],
        "product_categories": [],
        "key_search_terms": [],
    }


# ==============================================================================
# STEP 2: IDENTITY-POWERED PROMPT GENERATION
# ==============================================================================

async def auto_generate_brand_prompts(
    user_topic: str,
    domain: str,
    existing_queries: List[str] = None,
    location: str = "United States",
    max_count: int = 6,
    branded_count: int = 2,
    unbranded_count: int = 4,
    identity: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Identity-powered prompt generation. Returns a list of
    {"search_query", "topic", "prompt_type", "location"} dicts.

    Pass a pre-computed `identity` to skip re-research (the ADK agent
    researches once and reuses the identity across tools).
    """
    existing_queries = existing_queries or []
    year = datetime.now().year

    if identity is None:
        identity = await analyze_brand_identity(domain)

    brand_name = identity.get("brand_name", domain.split('.')[0].capitalize())
    industry = identity.get("industry", "Technology")
    specialty = identity.get("specialty", "Digital Products")
    biz_model = identity.get("business_model", "Online Business")
    target_audience = identity.get("target_audience", "General Consumers")
    pain_points = identity.get("audience_pain_points", [])
    competitors = identity.get("competitors", [])
    products = identity.get("product_categories", [])
    key_terms = identity.get("key_search_terms", [])

    pain_block = "\n".join(f"    - {p}" for p in pain_points[:5]) if pain_points else "    - (not available)"
    products_block = "\n".join(f"    - {p}" for p in products[:6]) if products else "    - (not available)"
    competitors_block = ", ".join(competitors[:5]) if competitors else "(unknown)"
    terms_block = "\n".join(f"    - \"{t}\"" for t in key_terms[:8]) if key_terms else "    - (not available)"

    exclude_block = ""
    if existing_queries:
        exclude_block = "\nDO NOT repeat these existing prompts:\n" + "\n".join(f"- {q}" for q in existing_queries[:50])

    prompt = f"""
You are a Search Query Strategist generating prompts for brand visibility monitoring.

═══ COMPANY PROFILE (researched from the web) ═══
Brand Name: "{brand_name}"
Domain: {domain}
Industry: {industry}
Specialty: {specialty}
Business Model: {biz_model}
Target Audience: {target_audience}

Customer Pain Points:
{pain_block}

Products/Services They Actually Offer:
{products_block}

Direct Competitors: {competitors_block}

Real Search Terms Their Audience Already Uses:
{terms_block}

USER'S REQUESTED TOPIC: "{user_topic}"
LOCATION: {location}

═══ TASK ═══
Generate exactly {max_count} search queries that REAL PEOPLE would type into Google,
ChatGPT, or Perplexity when looking for products/services like what "{brand_name}" offers.

== GROUP A: BRANDED QUERIES ({branded_count} total) ==
These MUST mention "{brand_name}" by name.
Patterns: "{brand_name} reviews {year}", "is {brand_name} worth it",
"{brand_name} vs {competitors[0] if competitors else 'competitors'} comparison".

== GROUP B: UNBRANDED / DISCOVERY QUERIES ({unbranded_count} total) ==
These must NOT mention "{brand_name}" at all. What would their TARGET AUDIENCE
({target_audience}) type when they have a PROBLEM but don't know this brand exists?
Angles: pain-point queries, product shopping queries, comparison queries,
how-to queries, best-of/roundup queries.

═══ CRITICAL RULES ═══
1. Every query must be 5-12 words, natural language
2. Unbranded queries MUST be specific to the "{industry}" niche
3. Include the year {year} in at least 2 queries
4. If user_topic is vague or generic, IGNORE IT and use the researched industry
5. All {max_count} queries must be meaningfully different
6. NEVER use generic filler phrases like "online services", "digital products",
   "technology company", "service providers"
{exclude_block}"""

    json_instruction = """
Return strict JSON:
{
  "prompts": [
    {"text": "the search query here", "type": "branded"},
    {"text": "another search query", "type": "unbranded"}
  ]
}
"""

    try:
        data = await robust_json_call(prompt, json_instruction, use_search=False)

        raw = data.get("prompts", [])
        seen = set(q.lower().strip() for q in existing_queries)
        result = []

        # Ground truth: detect branded from actual text, not the LLM's label
        brand_lower = brand_name.lower()
        domain_stem = domain.replace("https://", "").replace("http://", "").replace("www.", "").split('.')[0].lower()

        for item in raw:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            text = text.strip()
            if not text or len(text) < 10:
                continue
            if text.lower() in seen:
                continue

            text_lower = text.lower()
            generic_fillers = [
                "online services", "digital products", "technology company",
                "technology providers", "service providers", "general consumers",
                "online business", "web services",
            ]
            if any(filler in text_lower for filler in generic_fillers):
                continue

            mentions_brand = (brand_lower in text_lower) or (domain_stem in text_lower)
            result.append({
                "search_query": text,
                "topic": user_topic,
                "prompt_type": "branded" if mentions_brand else "unbranded",
                "location": location,
            })
            seen.add(text.lower())
            if len(result) >= max_count:
                break

        if not result:
            result.append({
                "search_query": f"{brand_name} {industry.lower()} reviews {year}",
                "topic": user_topic,
                "prompt_type": "branded",
                "location": location,
            })

        logger.info(f"✅ Generated {len(result)} prompts (brand: {brand_name}, industry: {industry})")
        return result

    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        fb_name = domain.split('.')[0].capitalize()
        return [{
            "search_query": f"{fb_name} reviews {year}",
            "topic": user_topic,
            "prompt_type": "branded",
            "location": location,
        }]
