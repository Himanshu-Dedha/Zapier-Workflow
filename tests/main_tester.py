"""
Manual test runner for individual modules WITHOUT modifying the app package.

Phases
------
1) sheets
   - Auth check
   - Read profiles & activities
   - Canonicalize/normalize columns
   - Validate required columns & data quality
   - Group activities and show join preview
   - (Optional) Write a temporary preview worksheet
   - (Optional) Mark 'processed' = TRUE for first N previewed URLs

2) llm
   - Same read/normalize/group as above
   - Filter to UNPROCESSED rows only (processed!=TRUE)
   - Join activities
   - Call LLM categorizer (LangChain: OpenAI or Google)
   - (Optional) Write an output worksheet with results
   - (Optional) Mark those processed urls as TRUE

Run from the project root (folder that contains the 'app/' package).

Examples
--------
# Sheets-only preview + mark processed for first 5:
python -m tests.main_tester --phase sheets --dry-write --mark-processed --limit 5

# LLM phase (OpenAI), write results, then mark processed for the rows handled:
python -m tests.main_tester --phase llm \
  --profiles-sheet "https://docs.google.com/spreadsheets/d/1A_KFeFWLA-dOfS9VGfDg-cDJvTRHBg_BuhEUO91BSrY/edit" \
  --activity-sheet "https://docs.google.com/spreadsheets/d/10pMMhjpPFq5o-yIMcynZsGFvYlAkHb4ih4SH5DVoLNU/edit" \
  --write-output --output-worksheet "Categorized_TEST" \
  --limit 10 \
  --mark-processed
"""

import argparse
from typing import Optional, Iterable, List, Dict, Any, Set, Tuple
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from app.models import Categorization
# App imports (do not modify app/*)
from app.config import settings
from app.sheets import (
    get_gspread_client,
    open_sheet,
    read_worksheet,
    to_dataframe,
    write_dataframe_to_new_worksheet,
)
from app.processing import group_activities_by_url, classify_record  # reuse LLM classifier
from app.llm import build_llm, build_chain


# ---------------- Small internal helpers (tester-only) ---------------- #

def _get_worksheet(spreadsheet, worksheet_name: Optional[str]):
    """Return gspread worksheet by name (or first sheet if None)."""
    return spreadsheet.worksheet(worksheet_name) if worksheet_name else spreadsheet.sheet1


def _ensure_column(ws, header_name: str) -> int:
    """
    Ensure a header column named `header_name` (case-insensitive) exists.
    If not present, append it at the end.
    Returns 1-based column index of that header.
    """
    headers = ws.row_values(1)
    lower = [h.strip().lower() for h in headers]
    name_lc = header_name.strip().lower()
    if name_lc in lower:
        return lower.index(name_lc) + 1  # 1-based
    # Append new header
    last_col = len(headers) + 1
    ws.update_cell(1, last_col, header_name)
    return last_col


def _find_column_index(ws, names: Iterable[str]) -> Optional[int]:
    """
    Find the first matching column index (1-based) for any of the candidate names (case-insensitive).
    """
    headers = ws.row_values(1)
    lower = [h.strip().lower() for h in headers]
    for n in names:
        nlc = n.strip().lower()
        if nlc in lower:
            return lower.index(nlc) + 1
    return None


def _mark_processed(ws, url_col_idx: int, processed_col_idx: int, urls_to_mark: Set[str]) -> int:
    """
    Batch-mark 'processed' = TRUE for rows whose URL is in urls_to_mark.
    Returns count of rows updated.
    """
    all_vals = ws.get_all_values()  # includes header row
    num_rows = len(all_vals)
    if num_rows <= 1:
        return 0

    # Build ranges for batch read/write (skip header at row 1)
    url_range = ws.range(2, url_col_idx, num_rows, url_col_idx)
    proc_range = ws.range(2, processed_col_idx, num_rows, processed_col_idx)

    updates = 0
    for url_cell, proc_cell in zip(url_range, proc_range):
        url_val = (url_cell.value or "").strip()
        if url_val and url_val in urls_to_mark:
            if (proc_cell.value or "").strip().upper() != "TRUE":
                proc_cell.value = "TRUE"
                updates += 1

    if updates:
        ws.update_cells(proc_range, value_input_option="USER_ENTERED")
    return updates


def _detect_processed_series(df: pd.DataFrame) -> pd.Series:
    """
    Return a boolean Series indicating processed==True based on a column named 'processed' (case-insensitive).
    If column missing, returns all False (i.e., treat all as unprocessed).
    """
    proc_col = None
    for c in df.columns:
        if str(c).strip().lower() == "processed":
            proc_col = c
            break
    if proc_col is None:
        return pd.Series(False, index=df.index)
    # normalize truthy values: TRUE/true/1/yes/y
    def as_bool(x):
        s = str(x).strip().lower()
        return s in {"true", "1", "yes", "y"}
    return df[proc_col].map(as_bool)


def _select_unprocessed_profiles(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter profiles_df to only rows where processed != True.
    """
    processed = _detect_processed_series(profiles_df)
    mask = ~processed.fillna(False)
    return profiles_df[mask].copy()


def _llm_rubric_text() -> str:
    """
    The rubric sent to the LLM. Edit as needed for your categories/rules.
    """
    return """
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
""".strip()


# ---------------- The test runner class ---------------- #

class ModuleTester:
    """
    Test runner for one module at a time. Supports:
      - Sheets phase
      - LLM phase
    """

    def __init__(self):
        """Load settings and prepare lazy resources."""
        self.settings = settings
        self._client = None
        # flags (set from CLI)
        self._mark_processed = False
        self._limit_to_mark = 10
        self._write_output = False
        self._output_worksheet = None
        self._max_items = None
        self._with_publications = False
        self._pubs_max = 5  

    # ---------- Connections ----------

    def _connect(self):
        """
        Authenticate to Google using service account from .env.
        Supports both FILE and INFO, whichever you set.
        """
        if self._client is None:
            sa_file = getattr(self.settings, "GOOGLE_SERVICE_ACCOUNT_FILE", None)
            sa_info = getattr(self.settings, "GOOGLE_SERVICE_ACCOUNT_INFO", None)
            self._client = get_gspread_client(sa_file, sa_info)
        return self._client

    def _open_two_sheets(
        self,
        profiles_sheet: str,
        activity_sheet: str,
        profiles_ws: Optional[str],
        activity_ws: Optional[str],
    ):
        """
        Open two spreadsheets and return (profiles_ss, acts_ss).
        """
        client = self._connect()
        return (
            open_sheet(client, profiles_sheet),
            open_sheet(client, activity_sheet),
        )

    # ---------- Validation helpers ----------

    def _validate_profiles_df(self, df: pd.DataFrame) -> None:
        """
        Validate presence of linkedin_url and print basic quality stats.
        """
        print("\n[Profiles] Columns:", list(df.columns))
        if "linkedin_url" not in df.columns:
            raise AssertionError("Profiles sheet must have a LinkedIn URL column (any variant; we normalize).")
        missing_urls = df["linkedin_url"].isna().sum()
        print(f"[Profiles] Rows: {len(df)}, Missing linkedin_url after normalization: {missing_urls}")

        print("[Profiles] Sample:")
        print(df.head(5).to_string(index=False))

    def _validate_activities_df(self, df: pd.DataFrame) -> None:
        """
        Validate presence of linkedin_url and print basic stats.
        """
        print("\n[Activities] Columns:", list(df.columns))
        if "linkedin_url" not in df.columns:
            raise AssertionError("Activities sheet must have a LinkedIn URL column (any variant; we normalize).")
        print(f"[Activities] Rows: {len(df)}")
        print("[Activities] Sample:")
        print(df.head(5).to_string(index=False))

    # ---------- Sheets phase ----------

    def run_sheets_phase(
        self,
        profiles_sheet: Optional[str],
        profiles_ws: Optional[str],
        activity_sheet: Optional[str],
        activity_ws: Optional[str],
        dry_write: bool = False,
    ) -> None:
        """
        Execute the Sheets-only test phase.
        """
        # Resolve params
        p_sheet = profiles_sheet or self.settings.PROFILES_SHEET_URL_OR_ID
        a_sheet = activity_sheet or self.settings.ACTIVITY_SHEET_URL_OR_ID
        p_ws = profiles_ws or self.settings.PROFILES_WORKSHEET
        a_ws = activity_ws or self.settings.ACTIVITY_WORKSHEET

        if not p_sheet or not a_sheet:
            raise RuntimeError("Missing profiles_sheet or activity_sheet. Set them via CLI or .env.")

        print("[Sheets] Authenticating…")
        profiles_ss, acts_ss = self._open_two_sheets(p_sheet, a_sheet, p_ws, a_ws)

        print("[Sheets] Reading worksheets…")
        profiles_rows = read_worksheet(profiles_ss, p_ws)
        acts_rows = read_worksheet(acts_ss, a_ws)
        print(f"[Sheets] Profiles rows: {len(profiles_rows)} | Activities rows: {len(acts_rows)}")

        print("[Sheets] Canonicalizing columns and normalizing URLs…")
        profiles_df = to_dataframe(profiles_rows)
        acts_df = to_dataframe(acts_rows)

        # Validate
        self._validate_profiles_df(profiles_df)
        self._validate_activities_df(acts_df)

        print("\n[Sheets] Grouping activities per linkedin_url…")
        grouped = group_activities_by_url(acts_df)
        print(f"[Sheets] Grouped activities rows: {len(grouped)}")
        print(grouped.head(5).to_string(index=False))

        print("\n[Sheets] Previewing left-join (profiles ⟕ activities)…")
        preview = profiles_df.merge(grouped, on="linkedin_url", how="left")
        keep = [c for c in ["full_name", "company_name", "title", "linkedin_url", "activities"] if c in preview.columns]
        print(preview[keep].head(10).to_string(index=False))

        if dry_write:
            print("\n[Sheets] Writing a temporary preview worksheet…")
            from datetime import datetime
            temp_name = f"_Z_TEST_PREVIEW_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            tiny = preview.head(10).copy()
            if "activities" in tiny.columns:
                tiny["activities"] = tiny["activities"].apply(
                    lambda x: " | ".join(x[:3]) if isinstance(x, list) else ""
                )
            write_dataframe_to_new_worksheet(profiles_ss, temp_name, tiny)
            print(f"[Sheets] Wrote preview to worksheet: {temp_name}")

        # Optional: mark processed in preview
        if self._mark_processed:
            print("\n[Sheets] Marking 'processed' = TRUE on profiles sheet for previewed URLs...")
            limit = int(self._limit_to_mark or 10)

            urls_to_mark: List[str] = []
            if "linkedin_url" in preview.columns:
                urls_to_mark = [
                    u for u in preview["linkedin_url"].head(limit).tolist()
                    if isinstance(u, str) and u.strip()
                ]
            unique_urls = set(urls_to_mark)
            if not unique_urls:
                print("[Sheets] No URLs to mark (preview empty or missing linkedin_url). Skipping.")
                return

            p_ws_obj = _get_worksheet(profiles_ss, p_ws)
            url_col_idx = _find_column_index(p_ws_obj, ["profileUrl", "linkedin_url", "profile url", "profile_url"])
            if not url_col_idx:
                print("[Sheets][ERROR] Could not find URL column (profileUrl/linkedin_url).")
                return

            processed_col_idx = _ensure_column(p_ws_obj, "processed")
            updated = _mark_processed(p_ws_obj, url_col_idx, processed_col_idx, unique_urls)
            print(f"[Sheets] Marked processed=TRUE for {updated} rows (limit={limit}).")

    # ---------- LLM phase ----------

    def run_llm_phase(
        self,
        profiles_sheet: Optional[str],
        profiles_ws: Optional[str],
        activity_sheet: Optional[str],
        activity_ws: Optional[str],
    ) -> None:
        """
        Execute the LLM categorization phase on UNPROCESSED rows only.
        Steps:
          1) Read + normalize Profiles and Activities
          2) Filter profiles to unprocessed only
          3) Join with grouped activities
          4) Build LLM chain and categorize
          5) (Optional) Write an output worksheet with results
          6) (Optional) Mark processed=TRUE for these URLs
        """
        # Resolve params
        p_sheet = profiles_sheet or self.settings.PROFILES_SHEET_URL_OR_ID
        a_sheet = activity_sheet or self.settings.ACTIVITY_SHEET_URL_OR_ID
        p_ws = profiles_ws or self.settings.PROFILES_WORKSHEET
        a_ws = activity_ws or self.settings.ACTIVITY_WORKSHEET
        out_ws = self._output_worksheet or self.settings.OUTPUT_WORKSHEET
        write_output = bool(self._write_output)

        if not p_sheet or not a_sheet:
            raise RuntimeError("Missing profiles_sheet or activity_sheet. Set them via CLI or .env.")

        print("[LLM] Authenticating & reading…")
        profiles_ss, acts_ss = self._open_two_sheets(p_sheet, a_sheet, p_ws, a_ws)
        profiles_rows = read_worksheet(profiles_ss, p_ws)
        acts_rows = read_worksheet(acts_ss, a_ws)

        # Canonicalize
        print("[LLM] Normalizing data…")
        profiles_df = to_dataframe(profiles_rows)
        acts_df = to_dataframe(acts_rows)

        # Filter unprocessed
        unprocessed_df = _select_unprocessed_profiles(profiles_df)
        if self._max_items:
            unprocessed_df = unprocessed_df.head(self._max_items)
        print(f"[LLM] Profiles total: {len(profiles_df)} | Unprocessed to handle now: {len(unprocessed_df)}")

        if unprocessed_df.empty:
            print("[LLM] Nothing to do (no unprocessed rows).")
            return

        # Group activities & join
        grouped = group_activities_by_url(acts_df)
        merged = unprocessed_df.merge(grouped, on="linkedin_url", how="left")
        def _to_text_activities(v):
            if isinstance(v, list):
                out = []
                for item in v:
                    if isinstance(item, dict):
                        # prefer postContent; fall back to any text-like field
                        txt = item.get("postContent") or item.get("activity") or item.get("content") or ""
                        if txt:
                            out.append(str(txt))
                    else:
                        out.append(str(item))
                return out
            return [str(v)] if pd.notna(v) and str(v).strip() else []

        merged["activities"] = merged["activities"].apply(_to_text_activities)


        # Build LLM chain
        provider = (self.settings.LLM_PROVIDER or "openai").lower()
        model = self.settings.LLM_MODEL
        temp = float(self.settings.LLM_TEMPERATURE or 0.1)
        llm = build_llm(provider=provider, model=model, temperature=temp)
        chain = build_chain(llm).bind(
            rubric=_llm_rubric_text(),
            format_instructions=JsonOutputParser(pydantic_object=Categorization).get_format_instructions(),
        )
        # Categorize
        print("[LLM] Categorizing…")
        records: List[Dict[str, Any]] = []
        from app.utils import first_nonempty  # reuse helper

        for _, row in merged.iterrows():
            profile = row.to_dict()
            activities = profile.get("activities") or []
            try:
                cat = classify_record(chain, profile, activities)
            except Exception as e:
                print(f"[LLM][WARN] LLM failed for {profile.get('linkedin_url')}: {e}")
                cat = None

            out = {
                "linkedin_url": profile.get("linkedin_url"),
                "full_name": first_nonempty(profile, ["full_name"]),
                "company_name": first_nonempty(profile, ["company_name"]),
                "title": first_nonempty(profile, ["title"]),
                "category": getattr(cat, "category", None) if cat else None,
                "confidence": getattr(cat, "confidence", None) if cat else None,
                "is_researcher": getattr(cat, "is_researcher", None) if cat else None,
                "is_scholar": getattr(cat, "is_scholar", None) if cat else None,   # new
                "description": getattr(cat, "description", None) if cat else None, # new
                "tags": ",".join(getattr(cat, "tags", [])) if cat else "",
                "reasons": getattr(cat, "reasons", None) if cat else None,
                "num_activities": len(activities),
                "activities_sample": " | ".join(activities[:5]),
            }
            is_researcher = bool(getattr(cat, "is_researcher", False)) if cat else False
            needs_lookup  = bool(getattr(cat, "needs_scholar_lookup", False)) if cat else False
            is_scholar    = bool(getattr(cat, "is_scholar", False)) if cat else False
            use_pubs      = is_scholar or is_researcher or needs_lookup

            publications_json = "[]"
            if self._with_publications and use_pubs:
                from app.publications import fetch_publications
                from app.utils import first_nonempty
                full_name = first_nonempty(profile, ["full_name"]) or ""
                company   = first_nonempty(profile, ["company_name"])
                try:
                    pubs = fetch_publications(full_name, company)
                    # trim for sheet
                    pubs = pubs[: self._pubs_max]
                    import json as _json
                    publications_json = _json.dumps([p.model_dump() for p in pubs], ensure_ascii=False)
                except Exception as e:
                    print(f"[LLM][WARN] Publications lookup failed for {full_name}: {e}")
            out.update({
                "needs_scholar_lookup": needs_lookup,
                "use_publications": use_pubs,
                "publications_json": publications_json,
            })

            records.append(out)

        results_df = pd.DataFrame(records)
        print("\n[LLM] Results preview:")
        print(results_df.head(10).to_string(index=False))

        # Optional write-out
        if write_output:
            print(f"[LLM] Writing results to worksheet: {out_ws}")
            write_dataframe_to_new_worksheet(profiles_ss, out_ws, results_df)

        # Optional mark processed = TRUE for the URLs we handled
        if self._mark_processed:
            print("[LLM] Marking processed=TRUE for categorized URLs…")
            urls = set(results_df["linkedin_url"].dropna().astype(str).tolist())
            if urls:
                p_ws_obj = _get_worksheet(profiles_ss, p_ws)
                url_col_idx = _find_column_index(p_ws_obj, ["profileUrl", "linkedin_url", "profile url", "profile_url"])
                if not url_col_idx:
                    print("[LLM][ERROR] Could not find URL column (profileUrl/linkedin_url).")
                else:
                    processed_col_idx = _ensure_column(p_ws_obj, "processed")
                    updated = _mark_processed(p_ws_obj, url_col_idx, processed_col_idx, urls)
                    print(f"[LLM] Marked processed=TRUE for {updated} rows.")
            else:
                print("[LLM] No URLs to mark (results empty).")


# ---------------- CLI ---------------- #

def main():
    parser = argparse.ArgumentParser(description="Manual tester for individual modules.")
    parser.add_argument("--phase", choices=["sheets", "llm"], default="sheets", help="Which module to test.")

    parser.add_argument("--profiles-sheet", default=None, help="Override profiles sheet URL/ID.")
    parser.add_argument("--profiles-worksheet", default=None, help="Override profiles worksheet name.")
    parser.add_argument("--activity-sheet", default=None, help="Override activities sheet URL/ID.")
    parser.add_argument("--activity-worksheet", default=None, help="Override activities worksheet name.")

    # Sheets phase helpers
    parser.add_argument("--dry-write", action="store_true", help="(sheets) Write a temporary preview worksheet.")

    # Shared options
    parser.add_argument("--mark-processed", action="store_true",
                        help="Mark 'processed' = TRUE for rows handled in this run.")
    parser.add_argument("--limit", type=int, default=None,
                        help="(sheets) How many preview rows to mark as processed. Also used as max_items in LLM phase.")

    # LLM output options
    parser.add_argument("--write-output", action="store_true",
                        help="(llm) Write results to an output worksheet.")
    parser.add_argument("--output-worksheet", default=None,
                        help="(llm) Results worksheet name (default from .env).")

        # in main(), add:
    parser.add_argument("--with-publications", action="store_true",
                        help="(llm) Fetch publications when flagged by the model.")
    parser.add_argument("--pubs-max", type=int, default=5,
                        help="(llm) Max pubs to keep per person in results (default 5).")

    args = parser.parse_args()

    tester = ModuleTester()
    tester._mark_processed = bool(args.mark_processed)
    tester._limit_to_mark = int(args.limit or 10)
    tester._write_output = bool(args.write_output)
    tester._output_worksheet = args.output_worksheet
    tester._max_items = args.limit  # reuse limit as max items for LLM
    tester._with_publications = bool(args.with_publications)
    tester._pubs_max = int(args.pubs_max or 5)  

    if args.phase == "sheets":
        tester.run_sheets_phase(
            profiles_sheet=args.profiles_sheet,
            profiles_ws=args.profiles_worksheet,
            activity_sheet=args.activity_sheet,
            activity_ws=args.activity_worksheet,
            dry_write=bool(args.dry_write),
        )
    elif args.phase == "llm":
        tester.run_llm_phase(
            profiles_sheet=args.profiles_sheet,
            profiles_ws=args.profiles_worksheet,
            activity_sheet=args.activity_sheet,
            activity_ws=args.activity_worksheet,
        )


if __name__ == "__main__":
    main()
