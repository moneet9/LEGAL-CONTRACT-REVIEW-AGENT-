# PS-9 — Legal Contract Review Agent

Local-first, evidence-grounded contract review with FastAPI, React, SQLite/filesystem persistence, local 768-dimensional embeddings, hybrid retrieval, selectable chat models, prompt-based review, audit logs, and human review. Gemini is optional; Ollama models are discovered when available.

## Quick start
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```
Then run `npm install && npm run dev` in `frontend`. The first upload downloads `sentence-transformers/all-mpnet-base-v2` into the local model cache. Copy `.env.example` to `.env` to enable Gemini chat, or run Ollama locally to expose installed models in the chat model picker. Runtime data is local under `data/`.

On Windows, use `start.cmd` to create the virtual environment, install dependencies, create `.env` if needed, and start both services. Use `stop.cmd` to stop them. The PowerShell equivalents are `start.ps1` and `stop.ps1`.

## Flow
```text
React → FastAPI → bounded contract prompt → structured findings → human review
```

POST a PDF to `/api/contracts`, then POST `{ "question": "Find financial risks" }` to `/api/contracts/{id}/review`. This is an engineering prototype, not legal advice.
