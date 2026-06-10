# Sanbi ADK Agent — Cloud Run image
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sanbi_core/ sanbi_core/
COPY agents/ agents/
COPY mcp_server/ mcp_server/

ENV PYTHONUNBUFFERED=1
ENV GOOGLE_GENAI_USE_VERTEXAI=TRUE

# Cloud Run injects $PORT. ADK api_server hosts the agent with a web UI.
CMD adk web agents --host 0.0.0.0 --port ${PORT:-8080}
