"""Local MCP server exposing the contract document classifier."""

from mcp.server.fastmcp import FastMCP

from .agents import classify_contract
from .internet_search import search_official_sources
from .models import Chunk

mcp = FastMCP("Contract Review Classifier", json_response=True)


@mcp.tool()
def classify_contract_document(document_text: str, filename: str = "") -> dict:
    """Identify the likely contract/document type using local rules and return a supporting quote.

    This tool does not store or transmit the supplied document text.
    """
    if not document_text or not document_text.strip():
        return {"document_type": "Other / unclear", "confidence": "low", "jurisdiction": "unknown",
                "alternatives": [], "reasoning": "No document text was supplied.", "evidence": None,
                "method": "local-keywords"}
    if len(document_text) > 100_000:
        raise ValueError("Document text is too long. Supply at most 100,000 characters.")
    chunk = Chunk(chunk_id="MCP-INPUT", contract_id="MCP-INPUT", page_start=1, page_end=1,
                  text=document_text, token_estimate=max(1, len(document_text) // 4))
    return classify_contract([chunk], document_name=filename)


@mcp.tool()
def search_official_legal_sources(query: str, max_results: int = 5) -> dict:
    """Search the internet for current material on official Indian government legal sites.

    Requires TAVILY_API_KEY in the project's .env. Results include source links and
    publication dates when available; verify the linked text and commencement date.
    """
    return search_official_sources(query, max_results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
