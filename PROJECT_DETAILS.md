# PS-9 Contract Review Agent — Project Details

## 1. What the project does

PS-9 is a local-first contract review workspace. Users can upload one or more PDF contracts, keep the original files, extract searchable evidence, generate embeddings, run a multi-agent review, and ask questions in a shared workspace chat.

The system is designed around evidence-grounded review:

1. A contract is uploaded and validated as a PDF.
2. The PDF is saved locally and assigned a workspace-scoped contract ID.
3. Text is extracted with PyMuPDF.
4. Extracted text is divided into clause-aware chunks.
5. Chunks are persisted as JSON.
6. Optional Gemini embeddings are saved as NumPy vectors and indexed with FAISS.
7. BM25 lexical retrieval and optional vector retrieval find relevant evidence.
8. Specialist agents analyze the evidence in parallel.
9. A cross-clause step, evidence verifier, and judge produce the final review.
10. Users can ask questions in a persistent workspace chat.

This is an engineering prototype and is not legal advice.

## 2. Main technologies

### Backend

- Python 3.10+
- FastAPI for the HTTP API
- Uvicorn for local serving
- Pydantic and pydantic-settings for models and configuration
- SQLite for metadata, findings, and chat message records
- Local filesystem for PDFs, chunks, embeddings, and indexes
- PyMuPDF (`fitz`) for PDF text extraction
- `rank-bm25` for lexical retrieval
- NumPy for vector storage and numerical operations
- FAISS CPU for vector similarity indexes
- LangGraph for explicit review workflow control flow
- Google Gemini API as an optional external model provider

### Frontend

- React
- TypeScript
- Vite
- `lucide-react` for interface icons
- CSS in `frontend/src/style.css`

## 3. Database and storage

The project uses SQLite. It does not currently use PostgreSQL, MongoDB, Redis, or a hosted vector database.

The database file is:

```text
data/metadata.db
```

SQLite tables:

### `contracts`

Stores one JSON-serialized `ContractMeta` record per contract.

Important fields include:

- `contract_id`
- `workspace_id`
- `original_filename`
- `sha256`
- `upload_time`
- `page_count`
- `chunk_count`
- `processing_status`
- `embedding_status`
- `analysis_status`

### `findings`

Stores review findings as JSON.

Fields include:

- finding ID
- contract ID
- serialized finding data
- human decision, when recorded

Supported human decisions are:

- `approved`
- `rejected`
- `more_evidence`
- `edited`

### `messages`

Stores contract-specific chat messages. This supports the individual contract chat API.

### `workspace_messages`

Stores the shared chat thread for one workspace. Messages are separated by `workspace_id`.

Each message stores:

- role: `user` or `assistant`
- message text
- timestamp
- serialized evidence citations

## 4. Filesystem storage layout

The data root is configured by `DATA_DIR` and defaults to `./data`.

```text
data/
├── metadata.db
├── contracts/
│   └── <contract_id>.pdf
├── chunks/
│   └── <contract_id>.json
├── embeddings/
│   └── <contract_id>.npy
├── indexes/
│   ├── <contract_id>.bm25.json
│   └── <contract_id>.faiss
├── audit/
│   └── events.jsonl
├── extracted/
├── pages/
├── reports/
└── cache/
```

The application creates all of these directories during startup. The active implementation writes PDFs, chunks, embeddings, indexes, and audit events. The other directories are reserved for future extracted-page, report, and cache artifacts.

## 5. Workspace isolation

The frontend creates a workspace ID in browser `sessionStorage` under:

```text
ps9-workspace-id
```

Every API request sends it as:

```http
X-Workspace-ID: <workspace-id>
```

The backend stores the workspace ID on every contract and filters workspace contract lists, document access, review access, activity, and workspace chat by that value.

Contract IDs include a short hash of the workspace ID and a short hash of the document content. This prevents the same PDF uploaded in two workspaces from overwriting the other workspace's files.

The shared workspace chat only accepts contract IDs that belong to the current workspace.

Existing data created before workspace isolation uses the `default` workspace.

## 6. Upload and indexing flow

### Step 1: PDF validation

The upload endpoint checks:

- MIME type is `application/pdf`
- File bytes begin with the PDF signature `%PDF`

Invalid files are rejected with HTTP 400.

### Step 2: Contract ID and checksum

The system computes a SHA-256 checksum of the original bytes.

The stored contract ID is based on:

```text
workspace hash + content hash
```

The original filename and SHA-256 digest are retained in the contract metadata.

### Step 3: PDF text extraction

PyMuPDF opens the PDF in memory and extracts page text.

Pages are represented internally as:

```text
(page_number, page_text)
```

Blank pages are ignored for chunk construction, but the total PDF page count is retained.

### Step 4: Clause-aware chunking

Text is split on blank-line paragraph boundaries. The ingester detects headings and clause numbers using patterns such as:

```text
1.
1.2
3(a)
```

Each chunk records:

- chunk ID
- contract ID
- starting page
- ending page
- section
- clause
- heading
- source text
- estimated token count

Chunks are limited to approximately 4,200 characters before a new chunk is started. Chunk JSON is saved to:

```text
data/chunks/<contract_id>.json
```

### Step 5: Embeddings

If `GEMINI_API_KEY` is configured, all chunk texts are sent to the configured Gemini embedding model:

```text
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
```

The returned vectors are saved as float32 NumPy arrays:

```text
data/embeddings/<contract_id>.npy
```

If Gemini is not configured, the contract remains usable with lexical retrieval and the metadata reports:

```text
fallback_no_gemini_key
```

## 7. Retrieval and indexing

The project uses hybrid retrieval.

### BM25 lexical retrieval

Every chunk is tokenized using word characters and lowercased. The token lists are used to build an in-memory BM25 index.

The token lists are also persisted to:

```text
data/indexes/<contract_id>.bm25.json
```

BM25 is useful for exact legal language such as:

- indemnification
- limitation of liability
- termination
- renewal
- confidentiality
- payment

### FAISS vector retrieval

When embeddings are present:

1. Chunk vectors are converted to float32.
2. Vectors are L2-normalized.
3. A FAISS `IndexFlatIP` index is created.
4. The index is populated with normalized chunk vectors.
5. The index is saved to:

```text
data/indexes/<contract_id>.faiss
```

The query is embedded with Gemini, normalized, and compared using inner product, which acts as cosine similarity after normalization.

### Combining results

The retriever obtains lexical candidates and optional semantic candidates. Duplicate chunk indexes are removed while preserving ranking order. The final top-K chunks are returned.

If no lexical or semantic result is available, a simple token-overlap fallback ranks chunks by shared query terms.

Configured retrieval limits are:

```text
TOP_K_BM25=30
TOP_K_VECTOR=30
TOP_K_RERANK=8
```

## 8. Multi-agent review workflow

The review is coordinated by `MasterOrchestratorAgent`.

### Planning

The orchestrator examines the user's review question and selects specialist agents by matching domain terms. If no domain terms match, all specialists are selected.

Available specialists:

- Liability Agent
- Payment Agent
- Termination Agent
- Privacy Agent
- IP Agent
- Compliance Agent

### Parallel specialist execution

Selected specialists run concurrently using Python `ThreadPoolExecutor`.

Each specialist:

1. Searches for its domain terms.
2. Selects relevant contract evidence.
3. Produces a grounded finding.
4. Validates that evidence is traceable to a source chunk.
5. Emits audit events for start, analysis, and completion.

If Gemini is configured, the specialist can generate a structured claim, reasoning, recommendation, severity, and confidence. Without Gemini, deterministic fallback wording is used.

### Cross-clause analysis

The cross-clause stage currently detects multiple payment findings and can produce a cross-clause finding when payment terms may conflict.

### Evidence verification

Findings without evidence are removed. Findings with evidence are marked:

```text
verification_status = verified
```

### Judge stage

The judge records the number of verified findings and passes the verified result to the human review gate.

The workflow is defined in `backend/app/graph.py`:

```text
plan
  ↓
specialists in parallel
  ↓
cross_clause
  ↓
verify
  ↓
judge
```

The graph can retry the specialist stage if verification does not pass, up to the configured retry limit.

## 9. Chat behavior

### Workspace chat

The workspace chat is the multi-contract conversation.

The frontend sends:

- user message
- list of contract IDs
- workspace ID through `X-Workspace-ID`

The backend:

1. Confirms every contract belongs to the current workspace.
2. Runs retrieval separately for each selected contract.
3. Combines the top evidence chunks.
4. Labels evidence with the source filename.
5. Uses Gemini for a structured answer when configured.
6. Falls back to the first relevant evidence chunk when Gemini is unavailable.
7. Saves both user and assistant messages in `workspace_messages`.

### Individual contract chat

The contract-specific endpoint remains available for single-document questions. It retrieves only the selected contract's chunks and stores messages in `messages`.

## 10. API endpoints

All workspace-sensitive requests should include:

```http
X-Workspace-ID: <workspace-id>
```

### Health

```http
GET /api/health
```

### Contracts

```http
POST /api/contracts
GET  /api/contracts
GET  /api/contracts/{contract_id}
GET  /api/contracts/{contract_id}/document
```

`POST /api/contracts` accepts multipart form data with the field name `file`.

### Review

```http
POST /api/contracts/{contract_id}/review
GET  /api/contracts/{contract_id}/review
```

Review request body:

```json
{
  "question": "Find financial, liability, privacy, and termination risks."
}
```

### Activity

```http
GET /api/contracts/{contract_id}/activity
```

### Contract chat

```http
GET  /api/contracts/{contract_id}/messages
POST /api/contracts/{contract_id}/chat
```

Request body:

```json
{
  "message": "What is the liability cap?"
}
```

### Workspace chat

```http
GET  /api/workspace/messages
POST /api/workspace/chat
```

Request body:

```json
{
  "message": "Compare the termination rights across these contracts.",
  "contract_ids": [
    "contract_<workspace>_<content>"
  ]
}
```

### Human decisions

```http
POST /api/findings/{finding_id}/decision
```

Request body:

```json
{
  "decision": "approved",
  "note": "Reviewed by counsel."
}
```

## 11. Configuration

Configuration is read from `.env`.

Important settings:

```env
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
DATA_DIR=./data
MAX_RETRIES=2
TOP_K_BM25=30
TOP_K_VECTOR=30
TOP_K_RERANK=8
```

Gemini is optional. Without an API key, uploads, chunking, BM25 retrieval, deterministic specialist review, and fallback chat remain available.

## 12. Running locally

### Backend

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
uvicorn backend.app.main:app --reload
```

The backend runs on:

```text
http://localhost:8000
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

The frontend runs on:

```text
http://localhost:5173
```

The repository also includes `start.cmd`, `start.ps1`, `stop.cmd`, and `stop.ps1` for local startup and shutdown.

## 13. Project structure

```text
backend/app/
├── config.py          Settings and environment configuration
├── gemini.py          Gemini generation and embedding client
├── graph.py           LangGraph review workflow
├── ingestion.py       PDF extraction and chunking
├── main.py            FastAPI routes
├── models.py          Pydantic request, response, and domain models
├── orchestrator.py    Planning, parallel agents, verification, judge
├── retrieval.py       BM25 + FAISS hybrid retrieval
├── security.py        Evidence and untrusted-content checks
├── specialists.py     Domain specialist agents
└── storage.py         SQLite and filesystem persistence

frontend/src/
├── main.tsx           React application and API interactions
└── style.css          Workspace UI styling
```

## 14. Current limitations

- SQLite and local filesystem are intended for local or single-instance use.
- There is no authentication or per-user authorization layer.
- Workspace IDs are generated by the browser session and passed in a request header; a production deployment should bind them to authenticated user and thread records on the server.
- Gemini responses depend on API availability, model limits, and the configured data-handling terms.
- The fallback chat response is intentionally simple when Gemini is unavailable.
- PDF extraction quality depends on whether the PDF contains selectable text. Scanned image PDFs need OCR support, which is not currently included.
- The current specialist logic is a prototype and should be evaluated against a larger legal benchmark before production use.

## 15. Data flow summary

```text
Browser workspace
      │  X-Workspace-ID
      ▼
FastAPI upload
      │
      ├── SQLite contract metadata
      ├── Local original PDF
      ├── PyMuPDF extraction
      ├── Clause-aware JSON chunks
      ├── Optional Gemini embeddings
      ├── BM25 token index
      └── Optional FAISS vector index

User review question
      │
      ▼
Master Orchestrator
      │
      ├── Planner
      ├── Parallel specialist agents
      ├── Cross-clause analysis
      ├── Evidence verifier
      └── Judge / human review gate

Workspace chat question
      │
      ▼
Per-contract retrieval → merged evidence → grounded answer → workspace message history
```
