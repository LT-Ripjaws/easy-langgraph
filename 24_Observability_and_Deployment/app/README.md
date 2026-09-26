# Deployable Chatbot Agent

A minimal LangGraph app in the shape LangSmith Deployment (formerly LangGraph Platform) expects: a compiled graph exported as a module-level variable, plus a `langgraph.json` config pointing at it.

## Files

- `agent.py` — a tool-calling chatbot graph (Gemini + a calculator tool), compiled and exported as `graph`.
- `langgraph.json` — tells the LangGraph CLI where to find dependencies, the graph, and environment variables.
- `README.md` — this file.

## Running Locally

From this folder, with the repo's virtual environment active:

```bash
langgraph dev --no-browser
```

This starts a local API server (default `http://127.0.0.1:2024`). Without `--no-browser`, it also opens LangSmith Studio in your browser. `GET /ok` confirms the server is up; `POST /assistants/search` lists the graphs it loaded (should include `agent`).

## Environment

`langgraph.json` points `"env"` at `../../.env`, the repo's root `.env` file. It needs a `GOOGLE_API_KEY` (see `.env.example` at the repo root). No LangSmith API key is required to run this app; tracing is optional and configured separately (see the guide in `24_Observability_and_Deployment/`).

## Building for Production

```bash
langgraph build -t my-agent-image
```

builds a Docker image. `langgraph up` runs that image locally with Docker Compose. Deploying it for real goes through LangSmith Deployment; see the guide one folder up for details.
