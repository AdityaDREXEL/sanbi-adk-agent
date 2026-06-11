"""
sanbi_core/execution.py — Multi-engine query execution.

Adapted from production Sanbi `core/execution.py`.

Track 3 scope: two engines — OpenAI (web-search-grounded) and Vertex Gemini
(Google-Search-grounded). Production Sanbi additionally runs Perplexity
and Claude through the same interface.

Each engine runs INDEPENDENTLY using its own native capabilities, with
location simulation to bias results toward a target market.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List

from .llm import openai_client, OPENAI_MODEL
from .gemini import gemini_grounded_text

logger = logging.getLogger(__name__)

ENGINES = ["openai", "gemini"]


async def fetch_engine_response(engine: str, prompt: str, location: str = "United States") -> Dict[str, Any]:
    """
    Routes the prompt to the specific LLM API and extracts BOTH text and raw
    citation objects. Returns {"text": str, "citations": [{"url", "title"}]}.
    """
    localized_prompt = f"Context: I am a user located in {location}.\nQuestion: {prompt}"

    # ==========================================================================
    # 1. GEMINI (Vertex AI) — native Google Search grounding
    # ==========================================================================
    if engine == "gemini":
        try:
            text, grounding_sources = await gemini_grounded_text(localized_prompt, use_search=True)

            clean_citations = []
            seen_urls = set()

            # PRIMARY: grounding metadata (authoritative)
            for src in grounding_sources:
                url = src.get("url", "")
                title = src.get("title", "Source")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    clean_citations.append({"url": url, "title": title})

            # FALLBACK: URLs Gemini wrote in the text body
            matches = re.findall(r'-\s*(?:\[(.*?)\]\s*-\s*)?(https?://\S+)', text)
            for title, url in matches:
                url = url.strip(".,)")
                if url and "vertexaisearch.cloud.google.com" not in url and url not in seen_urls:
                    seen_urls.add(url)
                    clean_citations.append({
                        "url": url,
                        "title": title.strip() if title else "Source",
                    })

            return {"text": text, "citations": clean_citations}

        except Exception as e:
            logger.error(f"Gemini execution error: {e}")
            return {"text": f"Error: {e}", "citations": []}

    # ==========================================================================
    # 2. OPENAI (GPT) — independent web-search-grounded audit
    # ==========================================================================
    elif engine == "openai":
        if not openai_client:
            return {"text": "OpenAI API key missing.", "citations": []}
        try:
            system_instruction = (
                f"You are an AI Auditor simulating a user in {location}. "
                "Use current web information to answer the user's question. "
                "Recommend specific brands, clinics, products, or services by "
                "name (not just generic advice), and cite the sources you used."
            )

            resp = await openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
            )

            msg = resp.choices[0].message
            content = msg.content or ""

            found_citations = []
            seen = set()

            # PRIMARY: web-search annotations (authoritative grounding metadata,
            # populated by the *-search-preview models) — mirrors how the Gemini
            # branch trusts Google's grounding metadata before parsing text.
            anns = getattr(msg, "annotations", None)
            if isinstance(anns, (list, tuple)):
                for ann in anns:
                    uc = getattr(ann, "url_citation", None)
                    if uc is None:
                        continue
                    u = (getattr(uc, "url", "") or "").strip().strip(".,)")
                    if u and u not in seen:
                        seen.add(u)
                        title = (getattr(uc, "title", "") or "Source").strip()
                        found_citations.append({"url": u, "title": title})

            # FALLBACK: URLs written inline in the text (covers non-search models
            # and any sources not surfaced as structured annotations).
            # 1. Markdown links [Title](url)
            for title, url in re.findall(r'\[(.*?)\]\((https?://\S+?)\)', content):
                u = url.strip(".,)")
                if u not in seen:
                    seen.add(u)
                    found_citations.append({"url": u, "title": title.strip()})

            # 2. Parenthesised bare URLs (https://url.com)
            for url in re.findall(r'(?<!\[)\((https?://[^\s)]+)\)', content):
                u = url.strip(".,)")
                if u not in seen:
                    seen.add(u)
                    found_citations.append({"url": u, "title": "Mentioned in Text"})

            # 3. Raw URLs not captured above
            for url in re.findall(r'(?<![\[(])https?://[^\s)\]"]+', content):
                u = url.strip(".,)")
                if u not in seen:
                    seen.add(u)
                    found_citations.append({"url": u, "title": "Mentioned in Text"})

            return {"text": content, "citations": found_citations}

        except Exception as e:
            logger.error(f"OpenAI execution error: {e}")
            return {"text": f"[OpenAI Error] {e}", "citations": []}

    return {"text": f"Unknown engine: {engine}", "citations": []}


async def query_all_engines(prompt: str, location: str = "United States") -> Dict[str, Dict[str, Any]]:
    """
    Run one prompt across all configured engines in parallel.
    Returns {engine_name: {"text", "citations"}}.
    """
    results = await asyncio.gather(
        *[fetch_engine_response(e, prompt, location) for e in ENGINES],
        return_exceptions=True,
    )
    out = {}
    for engine, res in zip(ENGINES, results):
        if isinstance(res, Exception):
            out[engine] = {"text": f"Error: {res}", "citations": []}
        else:
            out[engine] = res
    return out
