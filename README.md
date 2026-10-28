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

The contract workflow begins automatically after upload. `MasterAgent` supervises
the review stages, records attempts in the local audit log, retries failed agent
calls according to `MAX_RETRIES`, and runs independent checks in parallel. Long
contracts are scanned in context-capped batches so every indexed passage is covered.

### Review stages and order

1. **Prepare and index the PDF.** `Ingestion` extracts page text and creates chunks.
   `DocumentConfigAgent` chooses per-document indexing and retrieval settings, then
   `ChunkOptimizer` recommends chunk size and overlap. The app applies those settings
   and builds lexical and, when enabled, vector indexes.
2. **Classify and research.** `ContractClassifierAgent` identifies the likely document
   type and jurisdiction. `GovernmentResearchAgent` then searches official Indian
   government legal sites through Tavily using that classification. Its search
   results are additional legal context, kept separate from contract evidence.
3. **Scan the whole contract.** `InitialScan` reviews all indexed chunks in bounded
   batches and produces contract-giver and contract-acceptor findings.
4. **Run the first parallel group.** After the initial scan, these six agents run
   concurrently:
   - `GraphBuilder`
   - `ContractStructurer`
   - `SignatureVerifier`
   - `IntentAgent`
   - `DeadlineAgent`
   - `ObligationAgent`
5. **Validate and check coverage.** `GraphValidator` validates the graph, then
   `GraphReviewerAgent` reports graph issues. Next, these three local checks run
   concurrently: `CoverageAgent`, `DocumentQualityAgent`, and `SensitiveDataAgent`.
6. **Review and score each perspective.** `TemplateReviewAgent` checks applicable
   Indian property deed checklists. For each party perspective, the app then runs
   `FixerAgent` → `CrossChecker` → `ContractScorer` in sequence. The giver and
   acceptor passes run one after the other.
7. **Save the results.** The app saves both party reports, evidence, graph data,
   classification, government search results, and review activity for the UI.

### Chat flow

Chat is separate from the upload review. For each question, `RetrieverAgent` finds
relevant contract passages using hybrid lexical and vector search. `GraphRetriever`
adds graph relations supported by those passages. `AnswerGenerator` answers with
citations when the retrieved evidence supports it, or falls back to source excerpts
when it cannot produce a grounded answer.

### Upload and review agents

| Agent | Purpose and work |
|---|---|
| **MasterAgent** | Coordinates the review workflow, supervises agent calls, applies retries, records status and output summaries, and runs independent checks in parallel. |
| **Ingestion** | Extracts PDF text and page information, coordinates indexing, saves searchable chunks and embeddings when enabled, and records processing status. |
| **DocumentConfigAgent** | Profiles the extracted document and chooses per-contract settings for chunking, embeddings, retrieval, review context, and retries. |
| **ChunkOptimizer** | Measures chunk lengths and recommends chunk size and overlap before indexes are built; the ingestion flow applies the result. |
| **ContractClassifierAgent** | Classifies the document type and likely jurisdiction, with confidence and a supporting quote. Uses the selected model when available and local keyword rules otherwise. |
| **GovernmentResearchAgent** | Searches official Indian government legal domains with the shared Tavily search implementation, which is also exposed as an MCP tool. Saves linked results and supplies them as separate context to the review. |
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

The stdio MCP server exposes two tools to an external MCP host:

- `classify_contract_document` classifies supplied text with local rules and returns
  a likely document type, confidence, alternatives, and supporting quote.
- `search_official_legal_sources` searches Indian government legal domains with
  Tavily and returns source links, snippets, and publication dates when available.

The in-app review pipeline uses the same classification and search implementations
directly; it does not connect to its own server over the MCP protocol. After
classification, the pipeline searches for relevant official material and passes
the results into the review as separate legal context. The search query contains
the document type and jurisdiction, not the contract text. Search results are leads:
check the linked law and commencement date before relying on a rule. Start the MCP
server from the project directory with:

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
