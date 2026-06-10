"""
sanbi_core/gemini.py — Gemini helper layer (grounded text + strict JSON).

Adapted from production Sanbi `core/gemini_helpers.py`.

Track 3 refactor notes:
  - Identical google-genai SDK calls; backend swapped to Vertex AI via
    sanbi_core.llm client init.
  - Google Search grounding works the same on Vertex (Tool(google_search=...)).
  - Production's per-process daily cost limiter removed — quota management
    moves to GCP-native quotas + budgets in the refactor.
"""

import asyncio
import json
import logging
import re

import httpx
from google.genai import types

from .llm import gemini_client, GEMINI_MODEL, GEMINI_GRADING_MODEL

logger = logging.getLogger(__name__)

# Matches bare domain titles like "ti.com", "electrichybridvehicletechnology.com"
_BARE_DOMAIN_RE = re.compile(r'^[\w][\w.-]*\.[a-z]{2,}$', re.IGNORECASE)


async def _resolve_one(client: httpx.AsyncClient, uri: str, title: str) -> str:
    """
    Follow a vertexaisearch redirect one hop to get the actual public URL.
    Falls back to https://title if the redirect fails.
    """
    try:
        r = await client.get(uri, follow_redirects=False, timeout=3.0)
        loc = r.headers.get("location", "")
        if loc and loc.startswith("http"):
            return loc
    except Exception:
        pass
    t = (title or "").strip()
    return f"https://{t}" if t and _BARE_DOMAIN_RE.match(t) else uri


async def _resolve_grounding_urls(sources: list) -> list:
    """
    Resolve vertexaisearch.cloud.google.com redirect URIs to public page URLs.
    Runs all resolutions in parallel; falls back to https://domain from title.
    """
    if not sources:
        return sources
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=False,
    ) as client:
        resolved = await asyncio.gather(
            *[_resolve_one(client, s["url"], s["title"]) for s in sources],
            return_exceptions=True,
        )
    result = []
    for src, r in zip(sources, resolved):
        url = r if isinstance(r, str) and r.startswith("http") else src["url"]
        result.append({"url": url, "title": src["title"]})
    return result


async def gemini_grounded_text(prompt: str, use_search: bool = True) -> tuple:
    """
    Generate rich, grounded text (Reasoning/Search phase).

    Returns:
        (text, grounding_sources) where grounding_sources is a list of
        {"url": str, "title": str} dicts extracted from Google's grounding
        metadata. These are the authoritative citation references.
    """
    if not gemini_client:
        logger.error("Gemini client not initialized.")
        return "", []

    try:
        tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None

        enhanced_prompt = prompt
        if use_search:
            enhanced_prompt += (
                "\n\n[SYSTEM INSTRUCTION]: You MUST cite your sources at the very end of your response "
                "in this strict format so they can be parsed programmatically:\n"
                "### SOURCES\n"
                "- [Page Title] - https://actual-url.com\n"
                "- [Page Title] - https://actual-url.com"
            )

        active_model = GEMINI_MODEL if use_search else GEMINI_GRADING_MODEL

        config_settings = types.GenerateContentConfig(
            tools=tools,
            response_mime_type="text/plain",
        )

        resp = await gemini_client.aio.models.generate_content(
            model=active_model,
            contents=enhanced_prompt,
            config=config_settings,
        )

        text = resp.text or ""

        # Extract structured citations from grounding metadata. The
        # grounding_chunks contain redirect URIs that resolve to actual pages.
        grounding_sources = []
        if use_search and resp.candidates and resp.candidates[0].grounding_metadata:
            md = resp.candidates[0].grounding_metadata
            seen_uris = set()
            chunks = getattr(md, "grounding_chunks", []) or []
            for chunk in chunks:
                if hasattr(chunk, "web") and chunk.web:
                    uri = chunk.web.uri or ""
                    title = chunk.web.title or "Source"
                    if uri and uri not in seen_uris:
                        seen_uris.add(uri)
                        grounding_sources.append({"url": uri, "title": title})

            if grounding_sources:
                grounding_sources = await _resolve_grounding_urls(grounding_sources)

        return text, grounding_sources

    except Exception as e:
        logger.error(f"Gemini grounded text failed: {e}")
        return "", []


async def gemini_force_json(text_context: str, json_schema_prompt: str) -> dict:
    """
    Format text context into strict JSON (Formatting phase).
    Retries once on parse failure.
    """
    if not gemini_client:
        return {}

    formatting_prompt = f"""
    You are a Data Extraction Engine.

    SOURCE MATERIAL:
    {text_context[:30000]}

    TASK:
    {json_schema_prompt}

    RULES:
    - Extract information ONLY from the SOURCE MATERIAL provided above.
    - Return STRICT JSON only. No markdown formatting.
    """

    config_settings = types.GenerateContentConfig(
        response_mime_type="application/json",
        tools=None,
    )

    for attempt in range(2):
        try:
            resp = await gemini_client.aio.models.generate_content(
                model=GEMINI_GRADING_MODEL,
                contents=formatting_prompt,
                config=config_settings,
            )
            raw_text = (resp.text or "").strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return json.loads(raw_text)
        except Exception as e:
            logger.warning(f"JSON formatting failed (attempt {attempt + 1}): {e}")
            await asyncio.sleep(0.5)

    return {}


async def robust_json_call(prompt: str, json_instruction: str, use_search: bool = True) -> dict:
    """Reasoning + JSON in one call (no search) or two-step (search → format)."""
    if not use_search:
        return await _single_json_call(prompt, json_instruction)

    grounded_text, _sources = await gemini_grounded_text(prompt, use_search=True)
    if not grounded_text:
        return {}
    return await gemini_force_json(grounded_text, json_instruction)


async def _single_json_call(prompt: str, json_instruction: str) -> dict:
    """Combines reasoning + JSON formatting in one Gemini call (no search)."""
    if not gemini_client:
        return {}

    combined_prompt = f"""{prompt}

RESPOND IN STRICT JSON FORMAT. No markdown, no code fences.
{json_instruction}"""

    config_settings = types.GenerateContentConfig(
        response_mime_type="application/json",
        tools=None,
    )

    for attempt in range(2):
        try:
            resp = await gemini_client.aio.models.generate_content(
                model=GEMINI_GRADING_MODEL,
                contents=combined_prompt,
                config=config_settings,
            )
            raw_text = (resp.text or "").strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            return json.loads(raw_text)
        except Exception as e:
            logger.warning(f"Single JSON call failed (attempt {attempt + 1}): {e}")
            await asyncio.sleep(0.5)

    return {}
