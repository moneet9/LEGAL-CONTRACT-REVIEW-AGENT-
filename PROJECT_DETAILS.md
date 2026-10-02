# PS-9 Contract Review Agent — Project Details

## 1. What the project does

PS-9 is a local-first contract review workspace. Users can upload one or more PDF contracts, keep the original files, extract searchable evidence, generate embeddings, run a prompt-based review, and ask questions in a shared workspace chat.

The system is designed around evidence-grounded review:

1. A contract is uploaded and validated as a PDF.
2. The PDF is saved locally and assigned a workspace-scoped contract ID.
3. Text is extracted with PyMuPDF.
4. Extracted text is divided into clause-aware chunks.
5. Chunks are persisted as JSON.
6. A local Sentence Transformers model creates configured 384-dimensional embeddings saved in a persistent local Chroma vector store.
7. BM25 lexical retrieval and optional vector retrieval find relevant evidence.
8. One bounded prompt analyzes the evidence and returns structured findings.
9. Exact-quote validation filters unsupported model output.
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
- Chroma for persistent local vector similarity search
- Sentence Transformers with `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` for local 384-dimensional embeddings
- Google Gemini API and optional Ollama models for chat generation

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
- `indexing_stage`, `indexing_progress`, and `indexing_error`
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
│   └── <contract_id>.bm25.json
├── audit/
│   └── events.jsonl
├── extracted/
├── pages/
├── reports/
└── cache/
```

The application creates all of these directories during startup. The active implementation writes PDFs, chunks, embeddings, BM25 indexes, Chroma data, metadata, and audit events. The extracted, pages, reports, and cache directories are reserved for future use.

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

### Duplicate upload reuse

If an upload has the same workspace-scoped content hash and filename as a completed contract, and its chunk and embedding artifacts still exist, the API returns the existing metadata without repeating PDF extraction, chunking, or embedding. A same-named PDF with different content is indexed normally.

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

Chunks use `CHUNK_SIZE_CHARS` from `.env` and repeat the final `CHUNK_OVERLAP_CHARS` characters in the next chunk. Chunk JSON is saved to:

```text
data/chunks/<contract_id>.json
```

### Step 5: Embeddings

All chunk texts are encoded locally with the configured Sentence Transformers model:

```text
EMBEDDING_MODEL=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
EMBEDDING_DIMENSION=384
ENABLE_LOCAL_EMBEDDINGS=true
CHUNK_SIZE_CHARS=900
CHUNK_OVERLAP_CHARS=120
```

The returned vectors are saved as float32 NumPy arrays:

```text
data/embeddings/<contract_id>.npy
```

The model is downloaded on first use and cached by Sentence Transformers. The metadata reports:

```text
local_saved
```

## 7. Retrieval and indexing

The project uses persistent local Chroma plus hybrid retrieval.

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

### Chroma vector retrieval

When embeddings are present, chunk vectors are stored in a persistent local Chroma collection under:

```text
data/chroma/
```

Queries use the same local 384-dimensional embedding model. No Gemini or other network reranker is used.

### Combining results

The retriever obtains BM25 exact-language candidates and Chroma semantic candidates, combines them with reciprocal-rank fusion, then applies a deterministic token-overlap and exact-clause boost. Duplicate chunk indexes are removed and the final top-K chunks are returned.

If no lexical or semantic result is available, a simple token-overlap fallback ranks chunks by shared query terms.

Configured retrieval limits are:

```text
TOP_K_BM25=30
TOP_K_VECTOR=30
TOP_K_RERANK=8
```

## 8. Prompt-based review workflow

The review endpoint sends one bounded prompt to the selected Gemini or Ollama model. The prompt asks the model to prioritize material risks, return at most 12 deduplicated findings, cite exact contract quotes, and return only the configured JSON schema.

The API validates every returned quote against the stored chunks before saving a finding. If no model is selected, the API returns an empty report instead of running a multi-agent fallback.

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
5. Uses the selected Gemini or Ollama model for a structured, evidence-grounded answer when configured.
6. Retries other currently available Google models if a selected Gemini model is unavailable or overloaded.
7. Returns the provider error as HTTP 502 and records it in the audit log when generation fails.
8. Uses a contract-grounded fallback checklist if generation returns an empty or generic refusal.
9. Saves both user and assistant messages in `workspace_messages`.

Chat prompts support normal questions, hypotheticals, and possible-breach scenarios. The lawyer prompt must identify the relevant obligation, state assumptions, distinguish possible risk from a confirmed legal conclusion, cite page and clause evidence, and recommend safer steps such as notice, consent, written approval, cure, and record preservation.

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

### Available generation models

```http
GET /api/models
```

Returns currently discoverable Gemini and Ollama models plus the selected default. Provider discovery failures are ignored so one unavailable provider does not hide models from another.

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
GEMINI_MODEL=gemini-flash-lite-latest
EMBEDDING_MODEL=sentence-transformers/multi-qa-MiniLM-L6-cos-v1
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32
EMBEDDING_MAX_TOKENS=256
ENABLE_LOCAL_EMBEDDINGS=true
CHUNK_SIZE_CHARS=900
CHUNK_OVERLAP_CHARS=120
OLLAMA_BASE_URL=http://localhost:11434
DATA_DIR=./data
MAX_RETRIES=2
REVIEW_CONTEXT_CHARS=50000
TOP_K_BM25=30
TOP_K_VECTOR=30
TOP_K_RERANK=8
```

Gemini is optional. When Ollama is running, its installed models are returned by `GET /api/models` and can be selected in chat. Without a configured chat model, the evidence-grounded fallback remains available.

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
├── gemini.py          Gemini/Ollama model discovery and generation
├── embeddings.py      Local Sentence Transformers embeddings
├── ingestion.py       PDF extraction and chunking
├── main.py            FastAPI routes
├── models.py          Pydantic request, response, and domain models
├── retrieval.py       BM25 + Chroma hybrid retrieval
├── security.py        Evidence and untrusted-content checks
├── storage.py         SQLite and filesystem persistence
└── vector_store.py    Persistent Chroma collections

frontend/src/
├── main.tsx           React application and API interactions
├── style.css          Workspace UI styling and fixed-pane layout
└── progress.css       Progress-state styling
```

## 14. Current limitations

- SQLite and local filesystem are intended for local or single-instance use.
- There is no authentication or per-user authorization layer.
- Workspace IDs are generated by the browser session and passed in a request header; a production deployment should bind them to authenticated user and thread records on the server.
- Gemini responses depend on API availability, model limits, and the configured data-handling terms.
- The fallback chat response is a deterministic, contract-grounded checklist; it is not a substitute for model analysis or legal advice.
- PDF extraction quality depends on whether the PDF contains selectable text. Scanned image PDFs need OCR support, which is not currently included.
- Health scores are explainable heuristics, not legal conclusions. They deduct points for finding severity and for protection gaps identified in the indexed text.

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
      ├── Local Sentence Transformers embeddings
      ├── BM25 token index
      └── Persistent local Chroma vector store

User review question
  │
  ▼
FastAPI review route
  │
  ├── Bounded contract-review prompt
  ├── Structured model response
  ├── Exact-quote evidence validation
  └── Findings and explainable health summary

Workspace chat question
      │
      ▼
Per-contract retrieval → merged evidence → grounded answer → workspace message history
```
