"""
Publication lookup via SerpAPI (Google Scholar) and Semantic Scholar.

Call `fetch_publications(full_name, company)` to get a small list of top publications.
"""

import os
from typing import List, Optional
import requests

from .models import Publication
from .utils import safe_get


def fetch_publications_via_serpapi(query: str, num: int = 5) -> List[Publication]:
    """
    Query Google Scholar via SerpAPI.

    Processing:
        - Requires SERPAPI_API_KEY in environment.
        - Uses engine=google_scholar.
        - Parses results into Publication objects.

    Args:
        query: Scholar query (e.g., "Jane Doe Stanford").
        num: Max results to return.

    Returns:
        List of Publication objects (possibly empty if missing key or no results).
    """
    key = os.getenv("SERPAPI_API_KEY")
    if not key:
        return []

    params = {
        "engine": "google_scholar",
        "q": query,
        "hl": "en",
        "api_key": key,
    }
    r = requests.get("https://serpapi.com/search.json", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = []
    items = data.get("organic_results") or data.get("scholar_results") or []
    for it in items[:num]:
        title = safe_get(it, "title")
        url = safe_get(it, "link")
        citations = safe_get(it, "inline_links", "cited_by", "total")
        if not title:
            continue
        results.append(
            Publication(
                title=title,
                year=None,
                url=url,
                citations=int(citations) if isinstance(citations, int) else None,
                source="serpapi",
            )
        )
    return results


def fetch_publications_via_semantic_scholar(name: str, affiliation: Optional[str], limit: int = 5) -> List[Publication]:
    """
    Query Semantic Scholar for an author's top papers by citations.

    Processing:
        - Optional S2_API_KEY in environment.
        - First searches for an author by 'name [affiliation]'.
        - Fetches top N papers (by citationCount).

    Args:
        name: Author full name.
        affiliation: Affiliation hint (e.g., company/university).
        limit: Max number of papers to return.

    Returns:
        List of Publication objects (possibly empty).
    """
    headers = {}
    s2_key = os.getenv("S2_API_KEY")
    if s2_key:
        headers["x-api-key"] = s2_key

    q = name if not affiliation else f"{name} {affiliation}"
    r = requests.get(
        "https://api.semanticscholar.org/graph/v1/author/search",
        params={"query": q, "limit": 1, "fields": "name,affiliations"},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("data"):
        return []
    author = data["data"][0]
    author_id = author.get("authorId")
    if not author_id:
        return []

    r2 = requests.get(
        f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers",
        params={
            "fields": "title,year,externalIds,url,citationCount",
            "limit": limit,
            "sort": "citationCount",
            "order": "desc",
        },
        headers=headers,
        timeout=30,
    )
    r2.raise_for_status()
    papers = r2.json().get("data", [])
    pubs: List[Publication] = []
    for p in papers[:limit]:
        pubs.append(
            Publication(
                title=p.get("title"),
                year=p.get("year"),
                url=p.get("url"),
                citations=p.get("citationCount"),
                source="semantic_scholar",
            )
        )
    return pubs


def fetch_publications(full_name: str, company: Optional[str]) -> List[Publication]:
    query = f"{full_name} {company or ''}".strip()
    pubs = fetch_publications_via_serpapi(query=query, num=5)
    if pubs:
        return pubs
    # Only try Semantic Scholar if a key exists
    if os.getenv("S2_API_KEY"):
        return fetch_publications_via_semantic_scholar(name=full_name, affiliation=company, limit=5)
    return []
