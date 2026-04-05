# GitLab Handbook + Direction GenAI Chatbot

End-to-end Groq chatbot implementation for the assignment brief using Firecrawl MCP for live retrieval.

## What this project includes

- Groq Compound chat with Firecrawl MCP tool use (live per query)
- No vector database build/index step
- Source-scoped answers from GitLab Handbook + Direction/Releases pages
- Source URL citations in responses
- Streamlit chat frontend
- Optional FastAPI backend API
- Clear setup, run, and deployment instructions

## Stack

- Python
- Groq API (Groq Compound with MCP tool calling)
- Firecrawl MCP server (queried at runtime)
- Firecrawl API key for MCP server access
- Streamlit UI
- FastAPI API

## Project structure

```text
app/
  streamlit_app.py
api/
  server.py
chatbot/
  settings.py
  rag_service.py
requirements.txt
.env.example
README.md
```

## 1) Setup

### Prerequisites

- Python 3.10+
- Node.js + npx (required to run Firecrawl MCP server process)
- Firecrawl API key
- Groq API key

### Install dependencies

```powershell
cd e:\Joveo
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configure environment

```powershell
copy .env.example .env
```

Set values in `.env`:

- `FIRECRAWL_API_KEY`
- `GROQ_API_KEY`
- Optional model override if needed

Recommended free-tier defaults:
- `GROQ_CHAT_MODEL=meta-llama/llama-4-scout-17b-16e-instruct`

Optional MCP process overrides:
- `FIRECRAWL_MCP_COMMAND=npx`
- `FIRECRAWL_MCP_ARGS=-y,firecrawl-mcp`
- `FIRECRAWL_MCP_TIMEOUT_SECONDS=120`

Windows fallback (if direct `npx` launch fails):
- `FIRECRAWL_MCP_COMMAND=cmd`
- `FIRECRAWL_MCP_ARGS=/c,npx,-y,firecrawl-mcp`

## 2) Run the chatbot

### Streamlit app (recommended)

```powershell
streamlit run app/streamlit_app.py
```

### FastAPI backend (optional)

```powershell
uvicorn api.server:app --reload --port 8000
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

## 3) How it works

For each user question:
1. Groq Llama receives the question and tool access to Firecrawl MCP.
2. The model calls Firecrawl MCP tools to discover and scrape relevant GitLab pages.
3. The model returns an answer constrained to those fetched sources.
4. The app extracts and displays cited source URLs.

No local vector DB or embedding index is built.

## 4) Suggested deployment

### Streamlit Community Cloud or Hugging Face Spaces

- Keep the app entrypoint: `app/streamlit_app.py`
- Ensure environment variables are set in deployment secrets:
  - `FIRECRAWL_API_KEY`
  - `GROQ_API_KEY`
- Ensure Node.js/npx is available in the runtime so MCP server command can launch.

## 5) Guardrails and transparency implemented

- Source-scope policy in system prompt (Handbook + Direction/Releases only)
- Explicit fallback when evidence is insufficient
- Prompt-injection resistance (ignore page-embedded instructions)
- Source URL citations shown in UI and API response

## 6) Assignment deliverables mapping

- Project documentation: this README + your Google Doc write-up
- GitHub repository: includes full source code and local run steps
- Chatbot core: implemented via Groq + Firecrawl MCP live retrieval
- Frontend/UI: Streamlit chat interface with follow-up support and error handling
- Public deployment: supported via Streamlit/HF with env var configuration

## Notes

- Since retrieval is live per query, response time and reliability depend on Firecrawl availability/rate limits.
- For heavy production traffic, consider adding caching and retries around MCP tool calls.

## Troubleshooting

- If you see an error about Node.js or MCP startup, install Node.js and ensure `node` and `npx` are available in PATH.
- If MCP still fails on Windows PowerShell, use the `cmd` fallback variables shown above.
