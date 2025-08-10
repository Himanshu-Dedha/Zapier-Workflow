"""
Core data processing: merging sheets, calling the LLM, fetching publications,
and writing results.

Exposes `process()` which orchestrates the full pipeline.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .models import Categorization, Publication
from .utils import first_nonempty
from .publications import fetch_publications
from typing import Any, Dict, Iterable, List, Optional, Union
import json
from langchain.schema.runnable import Runnable
from pydantic import ValidationError
try:
    from langchain_core.output_parsers import JsonOutputParser
except Exception:  # fallback for older LC versions
    from langchain.output_parsers import JsonOutputParser  # type: ignore

# If your Categorization model lives elsewhere, adjust the import:
from app.models import Categorization
def _expected_vars_of_chain(chain) -> set:
    """
    Best-effort extraction of input variable names the Runnable expects.
    Works across multiple LC versions.
    """
    # LangChain Runnable often exposes a Pydantic input_schema
    schema = getattr(chain, "input_schema", None)
    if schema is not None:
        try:
            fields = getattr(schema, "model_fields", {})
            if fields:
                return set(fields.keys())
        except Exception:
            pass
        # Older .schema() path (less reliable, but try)
        try:
            props = schema.schema().get("properties", {})
            if props:
                return set(props.keys())
        except Exception:
            pass

    # Other fallbacks some versions expose
    for attr in ("input_keys", "input_variables"):
        keys = getattr(chain, attr, None)
        if keys:
            try:
                return set(list(keys))
            except Exception:
                pass

    # Default to the minimum we always send
    return {"profile", "activities"}


def group_activities_by_url(activity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple activity rows into a list per LinkedIn URL.

    Processing:
        - If sees an obvious activity column in {'activity','activities','post','posts','content'},
          group that column into a list.
        - Otherwise, serialize entire rows (minus URL) to JSON strings and group those.

    Args:
        activity_df: DataFrame of activities (should contain 'linkedin_url').

    Returns:
        DataFrame with columns: ['linkedin_url', 'activities'] where 'activities' is a list.
    """
    if activity_df.empty:
        return pd.DataFrame(columns=["linkedin_url", "activities"])

    # Guess activity column
    activity_col = None
    for c in activity_df.columns:
        if c.strip().lower() in {"activity", "activities", "post", "posts", "content"}:
            activity_col = c
            break

    if not activity_col:
        def row_to_text(r):
            r = {k: v for k, v in r.items() if k != "linkedin_url"}
            return json.dumps(r, ensure_ascii=False)
        grouped = (
            activity_df.groupby("linkedin_url", dropna=False)
            .apply(lambda g: [row_to_text(r) for _, r in g.iterrows()])
            .reset_index(name="activities")
        )
        return grouped

    grouped = (
        activity_df.groupby("linkedin_url", dropna=False)[activity_col]
        .apply(lambda s: [x for x in s if str(x).strip()])
        .reset_index(name="activities")
    )
    return grouped



def _extract_activity_text(item: Any) -> Optional[str]:
    """
    Best-effort extraction of displayable text from a single activity record.
    Supports:
      - dicts with "postContent" (preferred) or common fallbacks
      - plain strings
    Returns a stripped string or None.
    """
    if item is None:
        return None
    if isinstance(item, str):
        s = item.strip()
        return s or None
    if isinstance(item, dict):
        # prefer LinkedIn-like fields first
        for key in ("postContent", "activity", "content", "text", "title", "summary"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # last resort: serialize small dicts
        try:
            s = json.dumps(item, ensure_ascii=False)
            return s if len(s) <= 500 else s[:497] + "..."
        except Exception:
            return None
    # generic fallback
    try:
        s = str(item).strip()
        return s or None
    except Exception:
        return None


def _activities_to_lines(activities: Optional[Iterable[Any]], limit: int = 20) -> List[str]:
    """
    Normalize a list of activities to a list of printable lines.
    Truncates to `limit` items, skipping empties.
    """
    if not activities:
        return []
    lines: List[str] = []
    for it in activities:
        txt = _extract_activity_text(it)
        if txt:
            lines.append(txt)
        if len(lines) >= limit:
            break
    return lines


def _profile_to_text(profile: Dict[str, Any]) -> str:
    """
    Build a compact, deterministic textual summary of the profile dict to feed the LLM.
    Pulls common fields if present; includes a JSON tail so the model can recover extra context.
    """
    full_name = (profile.get("full_name") or "").strip()
    title = (profile.get("title") or "").strip()
    company = (profile.get("company_name") or "").strip()
    description = (profile.get("description") or "").strip()
    url = (profile.get("linkedin_url") or profile.get("profileUrl") or "").strip()

    parts = []
    if full_name:
        parts.append(f"Name: {full_name}")
    if title:
        parts.append(f"Title: {title}")
    if company:
        parts.append(f"Company: {company}")
    if url:
        parts.append(f"LinkedIn: {url}")
    if description:
        parts.append(f"About: {description}")

    # Always include a compact JSON tail with all fields for recoverability.
    try:
        tail = json.dumps(profile, ensure_ascii=False)
        parts.append(f"RawProfileJSON: {tail}")
    except Exception:
        pass

    return "\n".join(parts)


def classify_record(
    chain: Runnable,
    profile: Dict[str, Any],
    activities: Optional[Iterable[Any]] = None,
    *,
    max_activities: int = 20,
) -> Categorization:
    """
    Run the LLM categorizer chain for a single record and parse into `Categorization`.

    The function now auto-injects `rubric` and `format_instructions` if the chain
    expects them (so it won't fail when they weren't pre-bound).
    """
    # ---- 1) Prepare prompt inputs
    profile_text = _profile_to_text(profile)
    activity_lines = _activities_to_lines(activities, limit=max_activities)
    activities_block = "\n- " + "\n- ".join(activity_lines) if activity_lines else "(no recent activity found)"

    # ---- 2) Build payload, detect extra required variables dynamically
    payload = {
        "profile": profile_text,
        "activities": activities_block,
    }

    expected = _expected_vars_of_chain(chain)
    # Add rubric if the prompt expects it
    if "rubric" in expected:
        # concise default rubric; feel free to tweak
        default_rubric = (
            "Output STRICTLY valid JSON with keys: "
            "category, confidence (0-1), is_researcher, is_scholar, needs_scholar_lookup, "
            "description, tags (3-7 strings), reasons. "
            "Decide from profile + activities. Return JSON only."
        )
        payload["rubric"] = default_rubric

    # Add format_instructions if the prompt expects it
    if "format_instructions" in expected:
        parser = JsonOutputParser(pydantic_object=Categorization)
        payload["format_instructions"] = parser.get_format_instructions()

    # ---- 3) Invoke chain
    try:
        result = chain.invoke(payload)
    except Exception as e:
        raise ValueError(f"LLM invocation failed: {e}") from e

    # ---- 4) Normalize result into Categorization
    if isinstance(result, Categorization):
        return result

    if isinstance(result, dict):
        try:
            return Categorization.model_validate(result)
        except Exception:
            text = result.get("text")
            if isinstance(text, str):
                try:
                    return Categorization.model_validate(json.loads(text))
                except Exception:
                    pass
            raise

    if isinstance(result, str):
        return Categorization.model_validate(json.loads(result))

    content = getattr(result, "content", None)
    if isinstance(content, str):
        return Categorization.model_validate(json.loads(content))

    raise ValueError(f"Unexpected LLM output type: {type(result)}; cannot parse into Categorization.")


def process(
    profiles_df: pd.DataFrame,
    acts_df: pd.DataFrame,
    chain,
    write_output: bool = False,
    output_sheet=None,
    output_worksheet_name: Optional[str] = None,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """
    Orchestrate the full pipeline: join, LLM, optional publications, and write results.

    Processing:
        1) Group activities by URL.
        2) Left-join profiles and activities.
        3) For each profile row:
            - Call LLM for Categorization
            - If researcher-like: fetch publications
            - Build flattened output record
        4) Optionally write a fresh worksheet with all results.

    Args:
        profiles_df: Canonicalized profiles DataFrame (must include 'linkedin_url').
        acts_df: Canonicalized activities DataFrame (must include 'linkedin_url').
        chain: Runnable LLM chain.
        write_output: If True, results are written to `output_sheet`.
        output_sheet: gspread Spreadsheet handle to write into (if write_output).
        output_worksheet_name: Name of the output worksheet to create.
        max_items: Limit number of processed profile rows (debug/testing).

    Returns:
        A DataFrame of flattened results, one row per profile.
    """
    grouped_acts = group_activities_by_url(acts_df)
    merged = profiles_df.merge(grouped_acts, on="linkedin_url", how="left")
    merged["activities"] = merged["activities"].apply(
        lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
    )

    if max_items:
        merged = merged.head(max_items)

    rows_out: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
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
            full_name = first_nonempty(profile, ["full_name"])
            company = first_nonempty(profile, ["company_name"])
            if full_name:
                try:
                    pubs = fetch_publications(full_name, company)
                except Exception as e:
                    print(f"[WARN] Publications lookup failed for {full_name}: {e}")

        out = {
            "linkedin_url": profile.get("linkedin_url"),
            "full_name": first_nonempty(profile, ["full_name"]),
            "company_name": first_nonempty(profile, ["company_name"]),
            "title": first_nonempty(profile, ["title"]),
            "category": getattr(cat, "category", None) if cat else None,
            "confidence": getattr(cat, "confidence", None) if cat else None,
            "is_researcher": getattr(cat, "is_researcher", None) if cat else None,
            "tags": ",".join(getattr(cat, "tags", [])) if cat else "",
            "reasons": getattr(cat, "reasons", None) if cat else None,
            "num_activities": len(activities),
            "activities_sample": " | ".join(activities[:5]),
            "publications_json": json.dumps([p.model_dump() for p in pubs], ensure_ascii=False),
            "raw_model_json": json.dumps(cat.model_dump() if cat else {}, ensure_ascii=False),
        }
        rows_out.append(out)

    out_df = pd.DataFrame(rows_out)

    # Optional write-back
    if write_output and output_sheet and output_worksheet_name:
        from .sheets import write_dataframe_to_new_worksheet
        write_dataframe_to_new_worksheet(output_sheet, output_worksheet_name, out_df)

    return out_df
