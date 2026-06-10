"""
sanbi_core — Business-logic layer of the Sanbi brand-visibility audit.

Adapted from the production Sanbi codebase (https://sanbi.ai) for the
Google for Startups AI Agents Challenge, Track 3 (Refactor for Google Cloud
Marketplace & Gemini Enterprise).

Changes from production:
  - All Gemini calls routed through Vertex AI (google-genai SDK with
    vertexai=True) instead of the consumer Gemini API.
  - Engine matrix reduced to OpenAI + Vertex Gemini (production also runs
    Perplexity + Claude).
  - Supabase persistence, Stripe billing, and auth stripped — pure logic.
"""
