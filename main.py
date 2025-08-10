# filename: categorize_linkedin.py
import argparse
import os
import json
import time
import re
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ---- Google Sheets (gspread) ----
import gspread
from google.oauth2.service_account import Credentials

# ---- LangChain / LLM providers ----
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import JsonOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema import BaseOutputParser

# OpenAI & Google wrappers (install: langchain-openai, langchain-google-genai)
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# ------------- Configuration Helpers -------------

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


def get_gspread_client() -> gspread.Client:
    """
    Create a gspread client from either a service account file or json in env.
    """
    sa_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    sa_info = os.getenv("GOOGLE_SERVICE_ACCOUNT_INFO")

    if sa_file and os.path.exists(sa_file):
        creds = Credentials.from_service_account_file(sa_file, scopes=SCOPES)
    elif sa_info:
        info = json.loads(sa_info)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        raise RuntimeError(
            "Google service account not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE "
            "or GOOGLE_SERVICE_ACCOUNT_INFO."
        )
    return gspread.authorize(creds)


def open_sheet(client: gspread.Client, identifier: str) -> gspread.Spreadsheet:
    """
    identifier: either a full URL or a spreadsheet ID.
    """
    if identifier.startswith("http"):
        return client.open_by_url(identifier)
    return client.open_by_key(identifier)


def read_worksheet(ss: gspread.Spreadsheet, worksheet_name: Optional[str]) -> List[Dict[str, Any]]:
    ws = ss.worksheet(worksheet_name) if worksheet_name else ss.sheet1
    rows = ws.get_all_records()
    return rows


# ------------- Data Utilities -------------

def _first_nonempty(d: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        v = d.get(c)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def normalize_linkedin_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    u = str(raw).strip()
    if not u:
        return None

    # Handle plain usernames like "linkedin.com/in/john" or "in/john"
    if not u.lower().startswith("http"):
        if u.lower().startswith("linkedin.com"):
            u = "https://" + u
        else:
            # e.g., "in/john-doe"
            u = "https://www.linkedin.com/" + u.lstrip("/")

    try:
        p = urlparse(u)
        netloc = p.netloc.lower().replace("www.", "")
        path = re.sub(r"/+$", "", p.path)  # strip trailing slash
        # drop query & fragment
        return f"https://{netloc}{path}"
    except Exception:
        return u.lower()


def to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Normalize columns for common LinkedIn URL variants
    rename_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in {"linkedin", "linkedin url", "linkedin_url", "profile url", "profile_url"}:
            rename_map[col] = "linkedin_url"
        elif lc in {"full name", "fullname", "name"}:
            rename_map[col] = "full_name"
        elif lc in {"company", "company name", "employer"}:
            rename_map[col] = "company_name"
        elif lc in {"title", "job title"}:
            rename_map[col] = "title"
        elif lc in {"description", "summary", "about"}:
            rename_map[col] = "description"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "linkedin_url" in df.columns:
        df["linkedin_url"] = df["linkedin_url"].map(normalize_linkedin_url)
    return df


def group_activities_by_url(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects activity_df to contain 'linkedin_url' and 'activity' (or similar).
    Will combine multiple rows per URL into a list.
    """
    if activity_df.empty:
        return pd.DataFrame(columns=["linkedin_url", "activities"])

    # Guess activity column
    activity_col = None
    for c in activity_df.columns:
        if c.strip().lower() in {"activity", "activities", "post", "posts", "content"}:
            activity_col = c
            break
    # If no obvious column, combine entire row (minus URL)
    if not activity_col:
        def row_to_text(r):
            r = {k: v for k, v in r.items() if k != "linkedin_url"}
            return json.dumps(r, ensure_ascii=False)
        grouped = (activity_df
                   .groupby("linkedin_url", dropna=False)
                   .apply(lambda g: [row_to_text(r) for _, r in g.iterrows()])
                   .reset_index(name="activities"))
        return grouped

    # Found one activity column: group lists
    grouped = (activity_df
               .groupby("linkedin_url", dropna=False)[activity_col]
               .apply(lambda s: [x for x in s if str(x).strip()])
               .reset_index(name="activities"))
    return grouped


# ------------- LLM Output Schema -------------

class Categorization(BaseModel):
    category: str = Field(..., description="High-level category label according to the rubric.")
    reasons: str = Field(..., description="Short rationale for the categorization.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    is_researcher: bool = Field(
        ..., description="True if the person appears to be a researcher/PhD or has academic publications."
    )
    needs_scholar_lookup: bool = Field(
        ..., description="True if we should query scholarly publications for this person."
    )
    tags: List[str] = Field(default_factory=list, description="Additional lightweight tags.")
    # You can add/modify fields here to reflect your rubric.


def build_llm(provider: str, model: Optional[str] = None, temperature: float = 0.1):
    provider = (provider or "").lower()
    if provider == "openai":
        model = model or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=temperature)
    elif provider in {"google", "gemini"}:
        model = model or "gemini-1.5-pro"
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:
        raise ValueError("provider must be 'openai' or 'google'.")


def build_chain(llm) -> Runnable:
    parser: BaseOutputParser = JsonOutputParser(pydantic_object=Categorization)

    # ======== EDIT THIS RUBRIC as needed ========
    default_rubric = """
You are a B2B lead categorization assistant.

Output STRICTLY valid JSON that matches the schema provided.

Classify each person based on their LinkedIn profile + recent activity.
Example categories (EDIT ME): "Founder/Executive", "Research Scientist/PhD", "Engineer/IC", "Recruiter", "Investor/VC", "Other".

Heuristics (EDIT ME):
- "is_researcher" if PhD, publications, 'Research', 'Scholar', 'Professor', or clear academic signals.
- "needs_scholar_lookup" if is_researcher is likely True OR unclear but academic signals present.
Confidence: 0 to 1.
Tags: 3-7 short tokens (e.g., "ml", "genai", "healthcare", "fintech", "devrel").

Return ONLY JSON. No markdown, no code fences.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert classifier. Follow instructions carefully."),
            (
                "user",
                (
                    "RUBRIC:\n{rubric}\n\n"
                    "PROFILE:\n{profile}\n\n"
                    "ACTIVITIES (most recent first if available):\n{activities}\n\n"
                    "JSON schema:\n{format_instructions}"
                ),
            ),
        ]
    )

    chain = prompt | llm | parser
    return chain.bind(rubric=default_rubric, format_instructions=JsonOutputParser(pydantic_object=Categorization).get_format_instructions())


# ------------- Publications Lookup -------------

class Publication(BaseModel):
    title: str
    year: Optional[int] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    source: str


def safe_get(d: Dict[str, Any], *path, default=None):
    cur = d
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p)
    return cur if cur is not None else default


def fetch_publications_via_serpapi(query: str, num: int = 5) -> List[Publication]:
    """
    Uses SerpAPI's Google Scholar engine. Requires SERPAPI_API_KEY.
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
    # Parse scholar_results if present
    items = data.get("organic_results") or data.get("scholar_results") or []
    for it in items[:num]:
        title = safe_get(it, "title")
        url = safe_get(it, "link")
        year = None
        # Sometimes displayed as "Cited by X" etc.
        citations = None
        cite_str = safe_get(it, "inline_links", "cited_by", "total")
        if isinstance(cite_str, int):
            citations = cite_str
        if not title:
            continue
        results.append(Publication(title=title, year=year, url=url, citations=citations, source="serpapi"))
    return results


def fetch_publications_via_semantic_scholar(name: str, affiliation: Optional[str], limit: int = 5) -> List[Publication]:
    """
    Uses the Semantic Scholar API (free). Optional S2_API_KEY.
    """
    headers = {}
    s2_key = os.getenv("S2_API_KEY")
    if s2_key:
        headers["x-api-key"] = s2_key

    # Find author first
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

    # Get top publications (by citations)
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
    """
    Try SerpAPI Google Scholar first (if key present), else Semantic Scholar.
    """
    query = f'{full_name} {company or ""}'.strip()
    pubs = fetch_publications_via_serpapi(query=query, num=5)
    if pubs:
        return pubs
    return fetch_publications_via_semantic_scholar(name=full_name, affiliation=company, limit=5)


# ------------- Inference Loop -------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def classify_record(chain: Runnable, profile: Dict[str, Any], activities: List[str]) -> Categorization:
    # Compose a compact profile blob
    display = {
        "full_name": _first_nonempty(profile, ["full_name", "Full Name"]),
        "company": _first_nonempty(profile, ["company_name", "Company", "Company Name"]),
        "title": _first_nonempty(profile, ["title"]),
        "description": _first_nonempty(profile, ["description", "summary", "about"]),
        "linkedin_url": profile.get("linkedin_url"),
        # include any other useful fields:
        "raw": profile,
    }
    activities_text = "\n- " + "\n- ".join(activities[:30]) if activities else "None"

    result: Categorization = chain.invoke(
        {
            "profile": json.dumps(display, ensure_ascii=False),
            "activities": activities_text,
        }
    )
    return result


def process(
    profiles_df: pd.DataFrame,
    acts_df: pd.DataFrame,
    chain: Runnable,
    write_output: bool = False,
    output_sheet: Optional[gspread.Spreadsheet] = None,
    output_worksheet_name: Optional[str] = None,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    grouped_acts = group_activities_by_url(acts_df)
    merged = profiles_df.merge(grouped_acts, on="linkedin_url", how="left")
    merged["activities"] = merged["activities"].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x]))

    records = []
    total = len(merged)
    if max_items:
        merged = merged.head(max_items)

    for idx, row in merged.iterrows():
        profile = row.to_dict()
        activities = profile.get("activities") or []
        try:
            cat = classify_record(chain, profile, activities)
        except Exception as e:
            cat = None
            print(f"[WARN] LLM failed for {profile.get('linkedin_url')}: {e}")

        pubs: List[Publication] = []
        is_researcher = bool(cat and (cat.is_researcher or cat.needs_scholar_lookup))
        if is_researcher:
            full_name = _first_nonempty(profile, ["full_name"])
            company = _first_nonempty(profile, ["company_name"])
            if full_name:
                try:
                    pubs = fetch_publications(full_name, company)
                except Exception as e:
                    print(f"[WARN] Publications lookup failed for {full_name}: {e}")

        out = {
            "linkedin_url": profile.get("linkedin_url"),
            "full_name": _first_nonempty(profile, ["full_name"]),
            "company_name": _first_nonempty(profile, ["company_name"]),
            "title": _first_nonempty(profile, ["title"]),
            "category": getattr(cat, "category", None) if cat else None,
            "confidence": getattr(cat, "confidence", None) if cat else None,
            "is_researcher": getattr(cat, "is_researcher", None) if cat else None,
            "tags": ",".join(getattr(cat, "tags", [])) if cat else "",
            "reasons": getattr(cat, "reasons", None) if cat else None,
            "num_activities": len(activities),
            "activities_sample": " | ".join(activities[:5]),
            "publications_json": json.dumps([p.dict() for p in pubs], ensure_ascii=False),
            "raw_model_json": json.dumps(cat.dict() if cat else {}, ensure_ascii=False),
        }
        records.append(out)

    out_df = pd.DataFrame(records)

    if write_output and output_sheet and output_worksheet_name:
        # Create or clear worksheet
        try:
            try:
                ws = output_sheet.worksheet(output_worksheet_name)
                output_sheet.del_worksheet(ws)
            except gspread.exceptions.WorksheetNotFound:
                pass
            ws = output_sheet.add_worksheet(title=output_worksheet_name, rows=str(max(1000, len(out_df) + 10)), cols="30")
            # Write headers + rows
            ws.update([out_df.columns.tolist()] + out_df.fillna("").values.tolist())
        except Exception as e:
            print(f"[WARN] Failed writing output worksheet: {e}")

    return out_df


# ------------- CLI -------------

def main():
    ap = argparse.ArgumentParser(description="Join two Google Sheets by LinkedIn URL, categorize with LLM, and fetch publications.")
    ap.add_argument("--profiles-sheet", required=True, help="Profiles spreadsheet URL or ID (contains enriched LinkedIn URLs).")
    ap.add_argument("--profiles-worksheet", default=None, help="Worksheet name (default: first).")
    ap.add_argument("--activity-sheet", required=True, help="Activities spreadsheet URL or ID (multiple rows per LinkedIn URL).")
    ap.add_argument("--activity-worksheet", default=None, help="Worksheet name (default: first).")

    ap.add_argument("--provider", choices=["openai", "google"], required=True, help="LLM provider.")
    ap.add_argument("--model", default=None, help="Model name (optional).")
    ap.add_argument("--temperature", type=float, default=0.1)

    ap.add_argument("--write-output", action="store_true", help="Write results back to a worksheet in the PROFILES sheet.")
    ap.add_argument("--output-worksheet", default="Categorized", help="Output worksheet name (default: Categorized).")

    ap.add_argument("--max", type=int, default=None, help="Process at most N rows (debugging).")
    
    args = ap.parse_args()

    # Sheets
    client = get_gspread_client()
    profiles_ss = open_sheet(client, args.profiles_sheet)
    acts_ss = open_sheet(client, args.activity_sheet)

    profiles_rows = read_worksheet(profiles_ss, args.profiles_worksheet)
    acts_rows = read_worksheet(acts_ss, args.activity_worksheet)

    profiles_df = to_dataframe(profiles_rows)
    acts_df = to_dataframe(acts_rows)

    # Ensure linkedin_url exists
    if "linkedin_url" not in profiles_df.columns:
        raise RuntimeError("Profiles sheet must contain a 'LinkedIn URL' column (any case/variant).")
    if "linkedin_url" not in acts_df.columns:
        raise RuntimeError("Activities sheet must contain a 'LinkedIn URL' column (any case/variant).")

    # LLM chain
    llm = build_llm(provider=args.provider, model=args.model, temperature=args.temperature)
    chain = build_chain(llm)

    # Process
    out_df = process(
        profiles_df,
        acts_df,
        chain,
        write_output=args.write_output,
        output_sheet=profiles_ss if args.write_output else None,
        output_worksheet_name=args.output_worksheet if args.write_output else None,
        max_items=args.max,
    )

    # Print a small preview
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
