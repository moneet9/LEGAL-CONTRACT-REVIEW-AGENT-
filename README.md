# PS-9 — Legal Contract Review Agent

Evidence-grounded contract review prototype with FastAPI, React, SQLite/filesystem persistence, local embeddings, hybrid retrieval, supervised analysis agents, a contract relationship graph, and an autonomous review workflow. Gemini models appear when a Gemini API key is configured; Ollama models are also discovered when running.

## Quick start
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```
Then run `npm install && npm run dev` in `frontend`. Before uploading, choose the contract review model in the upload panel; choose a separate model for chat in the composer. The first upload may download the configured sentence-transformer model. Copy `.env.example` to `.env` to enable Gemini, or run Ollama locally. Online legal-source lookup uses `TAVILY_API_KEY` from `.env`.

On Windows, use `start.cmd` to create the virtual environment, install dependencies, create `.env` if needed, and start both services. Use `stop.cmd` to stop them. The PowerShell equivalents are `start.ps1` and `stop.ps1`.

## How the agents work

```text
Upload → Ingestion → document settings and indexing → document classification
                                          ↓
                                InitialScan + contract checks
                                          ↓
                             graph validation and review
                                          ↓
                 both party reports → evidence checks → scores

Chat question → RetrieverAgent → retrieved contract chunks
                                  + matching graph relations
                                  → AnswerGenerator → cited answer or abstention
```

`MasterAgent` supervises the review agents, records each attempt in the local audit
log, retries failures according to `MAX_RETRIES`, and runs independent checks in
parallel. Upload starts indexing and review automatically. Long contracts are sent
to review in context-capped batches so every indexed passage is covered.

### Upload and review agents

| Agent | Purpose and work |
|---|---|
| **MasterAgent** | Coordinates the review workflow, supervises agent calls, applies retries, records status and output summaries, and runs independent checks in parallel. |
| **Ingestion** | Extracts text and page information from the PDF, applies the finalized chunk settings, builds searchable indexes and local embeddings when available, and records indexing status. |
| **DocumentConfigAgent** | Profiles the extracted document and chooses per-contract settings for chunking, embeddings, retrieval, review context, and retries. |
| **ContractClassifierAgent** | Classifies the document type and likely jurisdiction, with confidence and a supporting quote. |
| **GovernmentResearchAgent** | Uses the MCP server's Tavily search tool to find related government legal sources after classification; saves results and supplies them as separate context to the review. |
| **ChunkOptimizer** | Measures the extracted text and recommends chunk size and overlap before search indexes are built. |
| **InitialScan** | Reviews every contract passage for material risks. When a model is selected, each review batch returns both the **contract giver** and **contract acceptor** views together. The giver is the party providing the stated goods, services, property, or draft; roles are inferred only from contract wording. Local mode uses the same role-neutral checklist for both views. |
| **GraphBuilder** | Sends the complete contract text in one request to extract named parties, terms, clauses, duties, dates, amounts, and supported relationships. It verifies returned quotes against page text and records their evidence. Without a usable model result, it builds a conservative local graph without exposing storage chunks as graph entities. |
| **GraphValidator** | Checks that graph links point to existing nodes and removes invalid, self-referential, or duplicate links. |
| **GraphReviewerAgent** | Reports disconnected entities and malformed or missing graph relationships for the agent activity view. |
| **ContractStructurer** | Groups extracted passages into sections and clauses, preserving page and chunk references for navigation and evidence. |
| **SignatureVerifier** | Finds signature, seal, witness, and signatory-authority signals. These are review leads; text inspection cannot authenticate a signature or prove authority. |
| **IntentAgent** | Flags suspicious instruction-like or hidden-intent text in the document. |
| **DeadlineAgent** | Extracts dates and checks their surrounding language for deadline signals and ambiguity. |
| **ObligationAgent** | Extracts duties expressed with terms such as “shall” and “must” and flags duties without a clearly assigned party. |
| **CoverageAgent** | Checks whether expected legal and commercial clause families appear in the contract. |
| **DocumentQualityAgent** | Looks for extraction problems such as OCR-like noise, repeated text, or empty pages. |
| **SensitiveDataAgent** | Locally detects patterns such as emails, phone numbers, account-like identifiers, and exposed secrets. It reports signals, not proof of a data breach. |
| **TemplateReviewAgent** | Compares likely Indian property deeds against the bundled deed checklists and reports present clauses, gaps, and red flags. It reports when the document is too short or the templates are unavailable. |
| **FixerAgent** | Removes findings that lack indexed evidence and repairs invalid graph relationships before the report is saved. |
| **CrossChecker** | Performs a final evidence and duplicate check on each party’s findings. A finding is retained only when its quoted evidence maps to an indexed chunk. |
| **ContractScorer** | Calculates an explainable contract health score for each party view from the findings and clause coverage. |

### Chat retrieval agents

| Agent | Purpose and work |
|---|---|
| **RetrieverAgent** | Searches the selected contract or contracts with the hybrid lexical and vector index and returns relevant evidence chunks. |
| **GraphRetriever** | Connects those retrieved chunks to graph relationships supported by the same page evidence. It supplies only the matching relations alongside the retrieved chunks. |
| **AnswerGenerator** | Answers from the supplied chunks and graph facts, cites supporting chunk IDs, and says when the retrieved evidence cannot support an answer. If a model answer cannot be grounded, the app falls back to source excerpts. |

The right-side perspective tabs switch between the already-saved giver and acceptor
reports; switching tabs does not start another model review. The chat answer remains
grounded in retrieved contract passages, with graph relations used as additional
context rather than as a replacement for cited source text.

### MCP tools

The MCP server exposes two tools: `classify_contract_document` uses local rules
and returns a likely document type with a supporting quote; `search_official_legal_sources`
uses Tavily to search current material on official Indian government legal sites.
The contract review pipeline runs classification first, then passes the search
results into the review as separate legal context. Results include links and
publication dates when available; confirm the linked law and commencement date.
Search queries contain the classified document type and jurisdiction, not the
contract text. Start the server from the project directory with:

```bash
python -m backend.app.mcp_server
```

Configure an MCP host to launch that command using the project's Python environment.
The `mcp` dependency is included in `backend/requirements.txt`. Without a Tavily key,
classification remains available and online lookup reports that it could not run.

Useful diagnostics endpoints:

- `GET /api/health` — service and orchestrator health.
- `GET /api/agents/telemetry` — per-agent runs, retries, failures, and recent events.
- `GET /api/contracts/{id}/graph` — locally persisted graph, structure, and signature signal.

Upload a PDF to `/api/contracts` as multipart form data with `file`. An optional `model`
may be supplied; otherwise the API selects the configured default model and falls back
to the local rule review when no model provider is available. Indexing queues the full
review automatically, so there is no manual analysis step. This remains an engineering
prototype, not legal advice.
