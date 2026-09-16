# PS-9 — Legal Contract Review Agent

Local-first, evidence-grounded contract review with FastAPI, React, SQLite/filesystem persistence, hybrid retrieval, explicit agent routing, verification, audit logs, and human review. Gemini is optional for local demonstration and is the only external AI provider when configured.

## Quick start
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```
Then run `npm install && npm run dev` in `frontend`. Copy `.env.example` to `.env` to enable Gemini. Runtime data is local under `data/`; uploaded content sent to Gemini is subject to applicable Gemini API data-handling terms.

On Windows, use `start.cmd` to create the virtual environment, install dependencies, create `.env` if needed, and start both services. Use `stop.cmd` to stop them. The PowerShell equivalents are `start.ps1` and `stop.ps1`.

## Flow
```text
React → FastAPI → Master Orchestrator → selected specialists → hybrid retrieval
                         ↓
                cross-clause → verifier → deterministic judge → human gate
```

POST a PDF to `/api/contracts`, then POST `{ "question": "Find financial risks" }` to `/api/contracts/{id}/review`. This is an engineering prototype, not legal advice.
