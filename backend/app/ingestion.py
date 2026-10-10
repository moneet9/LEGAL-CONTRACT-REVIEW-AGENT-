import re
import fitz
from .config import settings
from .models import Chunk


CLAUSE_RE = re.compile(
    r"^(\d+(?:\.\d+)*(?:\([A-Za-z0-9]+\))?)[.)]?\s+([A-Za-z][^\n]{2,140})$"
)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _chunk(cid, number, records):
    text = "\n\n".join(record["text"] for record in records)
    clauses = list(dict.fromkeys(record["clause"] for record in records if record["clause"]))
    headings = list(dict.fromkeys(record["heading"] for record in records if record["heading"]))
    clause = clauses[0] if len(clauses) == 1 else ("; ".join(clauses) if clauses else "")
    heading = headings[0] if len(headings) == 1 else ("; ".join(headings) if headings else "")
    section = clause.split(".")[0] if clause else ""
    return Chunk(
        chunk_id=f"{cid}_C{number:04d}",
        contract_id=cid,
        page_start=min(record["page"] for record in records),
        page_end=max(record["page"] for record in records),
        section=section,
        clause=clause,
        heading=heading,
        text=text,
        token_estimate=max(1, len(text) // 4),
    )


def _split_paragraph(text):
    if len(text) <= settings.chunk_size_chars:
        return [text]
    pieces = [piece.strip() for piece in SENTENCE_RE.split(text) if piece.strip()]
    result = []
    current = ""
    for piece in pieces or [text]:
        if current and len(current) + len(piece) + 1 > settings.chunk_size_chars:
            result.append(current)
            current = ""
        current = f"{current} {piece}".strip()
    if current:
        result.append(current)
    return result


def extract_and_chunk(pdf, cid):
    if settings.chunk_overlap_chars >= settings.chunk_size_chars:
        raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")
    doc = fitz.open(stream=pdf, filetype="pdf")
    pages = [(index + 1, page.get_text("text").strip()) for index, page in enumerate(doc)]
    chunks = []
    current = []
    active_clause = active_heading = ""

    def flush():
        nonlocal current
        if current:
            chunks.append(_chunk(cid, len(chunks) + 1, current))
            current = []

    for page, text in pages:
        if not text:
            continue
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            normalized = paragraph.replace("\n", " ")
            match = CLAUSE_RE.match(normalized)
            if match and not (len(match.group(1)) == 4 and match.group(1).startswith("20")):
                active_clause, active_heading = match.group(1), match.group(2)[:140]
            for piece in _split_paragraph(normalized):
                record = {
                    "page": page,
                    "clause": active_clause,
                    "heading": active_heading,
                    "text": piece,
                }
                if current and sum(len(item["text"]) for item in current) + len(piece) + 2 > settings.chunk_size_chars:
                    previous_text = "\n\n".join(item["text"] for item in current)
                    previous_record = current[-1]
                    flush()
                    overlap = previous_text[-settings.chunk_overlap_chars:]
                    overlap = overlap[overlap.find(" ") + 1:] if " " in overlap else overlap
                    if overlap:
                        current.append({
                            "page": previous_record["page"],
                            "clause": previous_record["clause"],
                            "heading": previous_record["heading"],
                            "text": overlap,
                        })
                current.append(record)
    flush()
    return len(doc), chunks
