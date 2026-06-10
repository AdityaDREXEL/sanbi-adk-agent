# Sanbi ADK Agent 🔍

**AI brand-visibility audits as an agent.** Ask "How visible is sight360.com for LASIK surgery?" and get a competitive leaderboard of who ChatGPT and Gemini *actually* recommend — built on Google's Agent Development Kit, Vertex AI, and the Model Context Protocol.

> **Google for Startups AI Agents Challenge — Track 3: Refactor for Google Cloud Marketplace & Gemini Enterprise.**
> This repo is a refactor of [sanbi.ai](https://sanbi.ai)'s production audit engine (live SaaS, FastAPI + Supabase + Railway) onto Google Cloud-native agent infrastructure.

---

## What it does

Brands are losing discoverability as search shifts to AI assistants. Sanbi answers the new question: **"When someone asks an AI for a recommendation in my category, do I show up?"**

The agent runs a 3-step audit pipeline:

1. **`generate_audit_prompts`** — researches the brand with **Gemini + Google Search grounding** (Vertex AI), extracts identity (industry, audience, competitors), and generates realistic branded + unbranded buyer queries.
2. **`query_engines`** — fires every query at multiple AI engines in parallel (OpenAI + Vertex Gemini with grounded search), capturing raw responses and citations.
3. **`grade_responses`** — LLM-grades each response (visibility, rank, sentiment, competitors mentioned), computes weighted visibility scores, and builds a **competitive leaderboard** + gap analysis + executive summary.

The agent orchestrates these tools conversationally — raw multi-KB engine responses stay in a server-side audit store; only compact summaries flow through the agent's context.

## Architecture

```
                        ┌──────────────────────────────┐
  user ── ADK web UI ──▶│  sanbi_audit_agent (ADK)     │
                        │  model: gemini-2.5-flash     │
                        │  tools:                      │
                        │   1. generate_audit_prompts  │──▶ Vertex Gemini + Google Search grounding
                        │   2. query_engines           │──▶ OpenAI ∥ Vertex Gemini (parallel)
                        │   3. grade_responses         │──▶ Vertex Gemini (JSON grading)
                        └──────────────┬───────────────┘
                                       │ shares sanbi_core/
                        ┌──────────────▼───────────────┐
  any MCP client ──────▶│  MCP server (FastMCP)        │
  (Claude, Gemini CLI)  │  tool: run_visibility_audit  │
                        └──────────────────────────────┘
                                  deployed on Cloud Run
```

- **`sanbi_core/`** — the audit engine, ported from production: planning (brand research + prompt generation), execution (multi-engine querying), analysis (grading + leaderboard).
- **`agents/sanbi_audit/`** — ADK agent: conversational orchestration of the pipeline.
- **`mcp_server/`** — the same audit exposed as a Model Context Protocol tool, so any MCP-capable agent can embed Sanbi audits.

## Quickstart

```bash
# 1. Clone + install
git clone https://github.com/AdityaDREXEL/sanbi-adk-agent && cd sanbi-adk-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # fill in GOOGLE_CLOUD_PROJECT + OPENAI_API_KEY
gcloud auth application-default login

# 3. Verify clients
python scripts/smoke_test.py

# 4. Run the agent (ADK dev UI at http://localhost:8000)
adk web agents
```

Then chat: *"Audit sight360.com for LASIK surgery in Philadelphia"*.

### Run the MCP server

```bash
# stdio (Claude Desktop / MCP Inspector)
python -m mcp_server.server

# HTTP (Cloud Run style)
MCP_TRANSPORT=http MCP_PORT=8081 python -m mcp_server.server
```

## Deploy to Cloud Run

```bash
gcloud run deploy sanbi-adk-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=$PROJECT,GOOGLE_CLOUD_LOCATION=us-central1 \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest
```

## Tests

134 offline tests cover the scoring formula, LLM-output coercion (None/string ranks, fenced JSON), citation extraction + Google-redirect filtering, leaderboard aggregation, engine-failure isolation, the agent's audit-store flow, and the MCP tool contract. All LLM calls are mocked — the suite runs with zero credentials and zero API spend.

```bash
pip install -r requirements-dev.txt
pytest
```

## Tech

| Layer | Tech |
|---|---|
| Agent framework | Google **Agent Development Kit (ADK)** |
| LLM | **Gemini 2.5 Flash via Vertex AI** (agent brain, research, grading, grounded search) |
| Audited engines | OpenAI + Vertex Gemini |
| Protocol | **Model Context Protocol (MCP)** |
| Runtime | **Cloud Run** (containerized, serverless) |
| Grounding | Vertex AI Google Search tool |

## Marketplace & Gemini Enterprise roadmap

This refactor is step 1 of bringing Sanbi to Google Cloud Marketplace:

- **Marketplace listing** — containerized Cloud Run service with usage-based billing hooks.
- **Gemini Enterprise / Agentspace** — register the agent so enterprise marketing teams can invoke audits from their Google Workspace.
- **AlloyDB** — replace production Supabase persistence for audit history & trend tracking.
- **Identity Platform** — multi-tenant auth for agency use.
- **Scheduled audits** — Cloud Scheduler → Cloud Run jobs for weekly visibility tracking (production Sanbi already does this on Railway; the port is mechanical).

## Relationship to production

[sanbi.ai](https://sanbi.ai) runs this same pipeline in production across 4 engines (OpenAI, Gemini, Perplexity, Claude) with batch execution, Supabase persistence, citation-growth mining (1,000+ community opportunities per brand), and Stripe billing. This repo extracts the core audit loop, re-routes all Gemini traffic through **Vertex AI**, and rebuilds the interface as an **ADK agent + MCP tool** — the agent-native form factor of the product.

## License

MIT
