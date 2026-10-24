# Testing

## Backend

Create or activate the project virtual environment, install the backend
requirements, and run the focused test suite:

```powershell
python -m pip install -r backend/requirements.txt
python -m pytest backend/tests -q
```

The current backend tests cover security-sensitive request validation. The
sample contracts in [`testing contracts/`](testing%20contracts/) can be used
for manual upload and review checks.

## Frontend

Install the frontend dependencies and create a production build:

```powershell
Set-Location frontend
npm install
npm run build
```

## Manual contract review checks

1. Start the application with `start.cmd`.
2. Upload each sample PDF from [`testing contracts/`](testing%20contracts/).
3. Confirm indexing reaches a completed state and the contract appears in the
   workspace.
4. Check both contract perspective reports, evidence citations, graph details,
   agent activity, and grounded chat responses.
5. Upload a template-compatible property deed and verify the template review
   uses the bundled checklists in
   [`india_property_contract_templates/`](india_property_contract_templates/).

The bundled files are test fixtures and templates only. This project is an
engineering prototype and does not provide legal advice.
