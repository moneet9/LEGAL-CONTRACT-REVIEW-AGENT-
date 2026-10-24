# India property & private-contract templates

Sale deed, agreement to sell, rental / leave-and-licence, lease deed, gift deed,
power of attorney, will, mortgage deed.

## Read this first: there is no single "central government standard"
Property documents are governed by central Acts (Transfer of Property Act 1882,
Registration Act 1908, Indian Stamp Act 1899, Indian Succession Act 1925) but **stamp duty,
registration fees and many mandatory formalities are set by each State**. What official
formats do exist are published by States / NGDRS (e.g. Punjab sale-deed format on
ngdrs.gov.in; Delhi Revenue Dept. model deeds) - see `official_sources.md`.

So this pack = (a) a checklist per deed type built from the Acts + the structure of the
official NGDRS/Delhi formats, and (b) fill-in sample forms. It does NOT contain state stamp-duty
rates (they change often) - it only checks that stamp/e-stamp details and registration are stated.

## Contents
- `templates/*.json`  8 deed checklists: required clauses, red flags, sources
- `forms/blank/`      fill-in forms (.txt + .pdf), placeholders in [BRACKETS]
- `forms/filled_samples/`  same forms with dummy values (use these to test uploads; PDFs pass your %PDF check)
- `check_template.py` detects deed type, then reports missing clauses, red flags, match score
- `build_property_pack.py`, `forms_text.py`, `make_pack.py` - edit and rebuild

## Use
    python check_template.py deed.pdf
    python check_template.py deed.pdf --type sale_deed --json
In code: `from check_template import check_contract; check_contract(text, "templates")`
Verdicts: MATCHES_TEMPLATE / PARTIAL_MATCH / TEMPLATE_NOT_MATCHING. If detection confidence is
"low" the report says so and lists candidate types - ask the user to confirm the type.

## What the red flags catch
notarised-only / plain-paper transfers, "GPA sale" (Suraj Lamp v. State of Haryana, SC 2011),
"need not be registered", large cash consideration, stated under-valuation, gift deed with a price,
will with one witness or stamped like a conveyance, clog on redemption, unfilled blanks.

## Limits
- Keyword/regex checks prove a clause is mentioned, not that it is legally sound. Use your LLM step to judge wording.
- State rules differ (e.g. Maharashtra requires registration of leave-and-licence agreements; Model Tenancy Act limits
  apply only where a State has adopted it). Always confirm with the State Registration & Stamps department.
- English only. Scanned PDFs need OCR. This is a screening aid, not legal advice.
