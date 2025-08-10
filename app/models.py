"""
Pydantic data models for API I/O and LLM outputs.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    """
    Request payload for /run endpoint.

    All fields are optional; if omitted, server falls back to .env defaults.

    Attributes:
        profiles_sheet: URL or spreadsheet ID of the profiles sheet.
        profiles_worksheet: Worksheet name within the profiles spreadsheet.
        activity_sheet: URL or spreadsheet ID of the activities sheet.
        activity_worksheet: Worksheet name within the activities spreadsheet.
        write_output: If true, writes results into a worksheet in the profiles sheet.
        output_worksheet: Name for the result worksheet (will be recreated).
        max_items: If set, processes at most N profile rows (debug/testing).
        provider: LLM provider ('openai' or 'google'); overrides .env.
        model: LLM model name; overrides .env.
        temperature: LLM temperature; overrides .env.
    """
    profiles_sheet: Optional[str] = None
    profiles_worksheet: Optional[str] = None
    activity_sheet: Optional[str] = None
    activity_worksheet: Optional[str] = None
    write_output: bool = True
    output_worksheet: Optional[str] = None
    max_items: Optional[int] = None

    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None


class Publication(BaseModel):
    """
    Publication record for results from SerpAPI/Semantic Scholar.

    Attributes:
        title: Paper title.
        year: Publication year (if known).
        url: Canonical URL (if known).
        citations: Citation count (if available).
        source: 'serpapi' or 'semantic_scholar'.
    """
    title: str
    year: Optional[int] = None
    url: Optional[str] = None
    citations: Optional[int] = None
    source: str


class Categorization(BaseModel):
    """
    Strict schema that the LLM must return per profile.

    Attributes:
        category: High-level label per your rubric.
        reasons: Short rationale for the label.
        confidence: Float 0–1 indicating confidence.
        is_researcher: True if person appears to be researcher/PhD.
        needs_scholar_lookup: True if we should fetch publications.
        tags: Short tags like 'ml', 'fintech', etc.
    """
    category: str
    reasons: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_researcher: bool
    needs_scholar_lookup: bool
    is_scholar: bool = False
    description: Optional[str] = None
    tags: List[str] = []


class RunSummaryItem(BaseModel):
    """
    A flattened line of output per processed profile, for API previews.

    Attributes:
        linkedin_url: Canonicalized LinkedIn URL.
        full_name: Person's name (if provided).
        company_name: Company (if provided).
        title: Job title (if provided).
        category: Categorization result.
        confidence: Confidence score (0–1).
        is_researcher: From the model output.
        num_activities: Count of aggregated activities.
        publications: First few publications (if any).
    """
    linkedin_url: Optional[str]
    full_name: Optional[str]
    company_name: Optional[str]
    title: Optional[str]
    category: Optional[str]
    confidence: Optional[float]
    is_researcher: Optional[bool]
    num_activities: int
    publications: List[Publication] = []


class RunResponse(BaseModel):
    """
    Response payload for /run endpoint.

    Attributes:
        processed: Total rows processed.
        written_to_sheet: True if results were written to the output worksheet.
        output_worksheet: Name of the output worksheet (if written).
        preview: Up to first 5 summarized results for quick inspection.
    """
    processed: int
    written_to_sheet: bool
    output_worksheet: Optional[str]
    preview: List[RunSummaryItem] = []
