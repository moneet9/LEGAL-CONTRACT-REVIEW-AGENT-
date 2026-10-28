"""Internet search restricted to official Indian government legal sources."""
from datetime import date
from urllib.parse import urlparse

import httpx

from .config import settings

OFFICIAL_DOMAINS = ["indiacode.nic.in", "legislative.gov.in", "egazette.nic.in", "gov.in", "nic.in"]


def search_official_sources(query: str, max_results: int = 5) -> dict:
    if not settings.tavily_api_key:
        return {"status": "not_configured", "as_of": date.today().isoformat(), "query": query,
                "sources": [], "note": "TAVILY_API_KEY is missing; online source lookup did not run."}
    query = query.strip()[:500]
    if not query:
        return {"status": "invalid_query", "as_of": date.today().isoformat(), "query": "", "sources": [],
                "note": "Enter a search query."}
    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json={"query": query, "search_depth": "advanced", "max_results": max(1, min(8, max_results)),
                  "include_domains": OFFICIAL_DOMAINS, "include_domains_mode": "restrict",
                  "include_answer": False, "include_published_date": True, "topic": "general", "language": "en"},
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
        sources = []
        for item in data.get("results", []):
            url = str(item.get("url", ""))
            host = (urlparse(url).hostname or "").lower()
            if not any(host == domain or host.endswith("." + domain) for domain in OFFICIAL_DOMAINS):
                continue
            sources.append({"title": str(item.get("title", "Official source"))[:240], "url": url,
                            "snippet": str(item.get("content", ""))[:1500], "source_domain": host,
                            "published_date": item.get("published_date")})
        status = "complete" if sources else "no_official_results"
        return {"status": status, "as_of": date.today().isoformat(), "query": query, "sources": sources,
                "note": "Search results are leads. Check the linked official text and commencement details before treating an amendment as in force."}
    except Exception as error:
        return {"status": "unavailable", "as_of": date.today().isoformat(), "query": query, "sources": [],
                "note": f"Online source lookup failed ({type(error).__name__}); no current rule was verified."}


def research_contract_rules(classification: dict) -> dict:
    kind = classification.get("document_type", "contract")
    jurisdiction = classification.get("jurisdiction", "India")
    query = f"{kind} {jurisdiction} India applicable acts rules latest amendments commencement notifications"
    return search_official_sources(query)


def format_research_context(research: dict) -> str:
    if not research.get("sources"):
        return "No official-source search results were retrieved. Do not claim current law was checked."
    return "\n".join(f"- {s.get('title')} ({s.get('source_domain')}, published {s.get('published_date') or 'date unknown'}): {s.get('snippet')} URL: {s.get('url')}" for s in research["sources"])
