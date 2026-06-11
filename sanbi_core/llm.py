"""
sanbi_core/llm.py — LLM client initialization.

Adapted from production Sanbi `core/llm_client.py`.

Track 3 refactor: the Gemini client now routes through **Vertex AI** via the
google-genai SDK (`vertexai=True`). Same SDK surface as production, different
backend — which is exactly the migration story this submission demonstrates.

Auth on Cloud Run: ambient service-account credentials (no key files).
Auth locally:      `gcloud auth application-default login`.
"""

import os
import logging

from dotenv import load_dotenv
from google import genai
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

# ==============================================================================
# CLIENT INITIALIZATION
# ==============================================================================

# 1. Gemini via Vertex AI (reasoning core + grader + audit engine #2)
gemini_client = None
_use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"
try:
    if _use_vertex:
        gemini_client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        )
        logger.info("✅ Gemini Client initialized (Vertex AI)")
    elif os.getenv("GEMINI_API_KEY"):
        # Dev-only fallback so the repo runs without a GCP project.
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        logger.info("✅ Gemini Client initialized (API key — dev fallback)")
    else:
        logger.warning("⚠️ No Gemini credentials. Set GOOGLE_GENAI_USE_VERTEXAI=TRUE + GOOGLE_CLOUD_PROJECT.")
except Exception as e:
    logger.error(f"Failed to initialize Gemini client: {e}")

# 2. OpenAI (audit engine #1)
openai_client = None
if os.getenv("OPENAI_API_KEY"):
    try:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ OpenAI Client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
else:
    logger.warning("⚠️ OPENAI_API_KEY not set — OpenAI engine disabled.")

# ==============================================================================
# MODEL CONSTANTS
# ==============================================================================
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_GRADING_MODEL = os.getenv("GEMINI_GRADING_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-search-preview")
