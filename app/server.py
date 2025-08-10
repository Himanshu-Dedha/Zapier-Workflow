"""
FastAPI server exposing a /run endpoint for Zapier (or manual) triggers.

Security:
    - Requires 'X-API-Key' header to match SERVICE_API_KEY (.env).
Usage:
    - POST /run with optional payload to override .env defaults.
"""

from typing import List, Optional

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from langchain_core.output_parsers import JsonOutputParser
from .models import Categorization

from .config import settings
from .models import RunRequest, RunResponse, RunSummaryItem, Publication
from .sheets import (
    get_gspread_client,
    open_sheet,
    read_worksheet,
    to_dataframe,
)
from .llm import build_llm, build_chain
from .processing import process

app = FastAPI(title="LinkedIn Categorizer Service", version="1.0.0")


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    """
    Dependency to enforce API key auth for all protected endpoints.

    Args:
        x_api_key: Value from request header 'X-API-Key'.

    Raises:
        HTTPException(401): If key missing or incorrect.
    """
    if not x_api_key or x_api_key != settings.SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
def health() -> dict:
    """
    Lightweight health check endpoint.

    Returns:
        dict: {'status': 'ok'}
    """
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse, dependencies=[Depends(require_api_key)])
def run_pipeline(req: RunRequest) -> RunResponse:
    """
    Run the full pipeline synchronously.

    Processing:
        - Resolve effective parameters from request or .env defaults.
        - Authenticate to Google Sheets and read both worksheets.
        - Build LLM chain (OpenAI or Google).
        - Run processing (join, categorize, publications).
        - Optionally write results to a worksheet.
        - Return a concise JSON summary for Zapier.

    Args:
        req: RunRequest with optional overrides.

    Returns:
        RunResponse summarizing the job (counts + preview).
    """
    # Resolve effective params (request overrides env)
    profiles_sheet = req.profiles_sheet or settings.PROFILES_SHEET_URL_OR_ID
    profiles_ws = req.profiles_worksheet or settings.PROFILES_WORKSHEET
    activity_sheet = req.activity_sheet or settings.ACTIVITY_SHEET_URL_OR_ID
    activity_ws = req.activity_worksheet or settings.ACTIVITY_WORKSHEET
    output_ws = req.output_worksheet or settings.OUTPUT_WORKSHEET
    write_output = req.write_output

    provider = (req.provider or settings.LLM_PROVIDER or "openai").lower()
    model = req.model or settings.LLM_MODEL
    temperature = req.temperature if req.temperature is not None else settings.LLM_TEMPERATURE

    # Validate required values
    for name, val in [
        ("profiles_sheet", profiles_sheet),
        ("activity_sheet", activity_sheet),
    ]:
        if not val:
            raise HTTPException(status_code=400, detail=f"Missing required parameter: {name}")

    # Sheets I/O
    client = get_gspread_client(
        settings.GOOGLE_SERVICE_ACCOUNT_FILE,
        settings.GOOGLE_SERVICE_ACCOUNT_INFO,
    )
    profiles_ss = open_sheet(client, profiles_sheet)
    acts_ss = open_sheet(client, activity_sheet)

    profiles_rows = read_worksheet(profiles_ss, profiles_ws)
    acts_rows = read_worksheet(acts_ss, activity_ws)

    profiles_df = to_dataframe(profiles_rows)
    acts_df = to_dataframe(acts_rows)

    if "linkedin_url" not in profiles_df.columns:
        raise HTTPException(status_code=400, detail="Profiles sheet must contain a LinkedIn URL column.")
    if "linkedin_url" not in acts_df.columns:
        raise HTTPException(status_code=400, detail="Activity sheet must contain a LinkedIn URL column.")

    # LLM chain
    llm = build_llm(provider=provider, model=model, temperature=temperature)
    parser = JsonOutputParser(pydantic_object=Categorization)
    chain = build_chain(llm).bind(
    rubric="""
You are a B2B lead categorization assistant.

Output STRICTLY valid JSON that matches the schema provided.

Classify each person based on their LinkedIn profile + recent activity.
Example categories: "Founder/Executive", "Research Scientist/PhD", "Engineer/IC", "Recruiter", "Investor/VC", "Other".

Heuristics:
- "is_researcher" if PhD, publications, 'Research', 'Scholar', 'Professor', or clear academic signals.
- "needs_scholar_lookup" if is_researcher likely True OR unclear but academic signals present.
Confidence: 0 to 1.
Tags: 3–7 short tokens.

Return ONLY JSON. No markdown, no code fences.
""",
    format_instructions=parser.get_format_instructions(),   # ← not empty
)
    # Run processing
    out_df = process(
        profiles_df=profiles_df,
        acts_df=acts_df,
        chain=chain,
        write_output=write_output,
        output_sheet=profiles_ss if write_output else None,
        output_worksheet_name=output_ws if write_output else None,
        max_items=req.max_items,
    )

    # Build API preview (first 5 rows)
    preview: List[RunSummaryItem] = []
    for _, r in out_df.head(5).iterrows():
        pubs = []
        try:
            pubs_json = r.get("publications_json")
            if pubs_json:
                from json import loads
                pubs = [Publication(**p) for p in loads(pubs_json)]
        except Exception:
            pubs = []
        preview.append(
            RunSummaryItem(
                linkedin_url=r.get("linkedin_url"),
                full_name=r.get("full_name"),
                company_name=r.get("company_name"),
                title=r.get("title"),
                category=r.get("category"),
                confidence=r.get("confidence"),
                is_researcher=r.get("is_researcher"),
                num_activities=int(r.get("num_activities") or 0),
                publications=pubs[:3],
            )
        )

    return RunResponse(
        processed=len(out_df),
        written_to_sheet=bool(write_output),
        output_worksheet=output_ws if write_output else None,
        preview=preview,
    )
