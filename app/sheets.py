"""
Google Sheets I/O utilities using gspread.

Centralizes authentication and read/write helpers so you could
swap to Salesforce or a database later with minimal changes.
"""

import json
import os
from typing import Any, Dict, List, Optional

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

from .utils import normalize_linkedin_url


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]


def get_gspread_client(sa_file: Optional[str], sa_info: Optional[str]) -> gspread.Client:
    """
    Create an authenticated gspread client.

    Processing:
        - Uses service account file path if provided and exists.
        - Else uses inline JSON string from env.
        - Raises if neither is available.

    Args:
        sa_file: Path to service account key JSON file.
        sa_info: Inline JSON string of the service account key.

    Returns:
        Authorized gspread.Client.
    """
    creds: Credentials
    if sa_file and os.path.exists(sa_file):
        creds = Credentials.from_service_account_file(sa_file, scopes=SCOPES)
    elif sa_info:
        info = json.loads(sa_info)
        creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        raise RuntimeError(
            "Google service account not configured. Set GOOGLE_SERVICE_ACCOUNT_FILE "
            "or GOOGLE_SERVICE_ACCOUNT_INFO in your environment."
        )
    return gspread.authorize(creds)


def open_sheet(client: gspread.Client, identifier: str) -> gspread.Spreadsheet:
    """
    Open a spreadsheet given either a full URL or a spreadsheet ID.

    Args:
        client: Authorized gspread client.
        identifier: URL or spreadsheet ID.

    Returns:
        gspread.Spreadsheet handle.
    """
    if identifier.startswith("http"):
        return client.open_by_url(identifier)
    return client.open_by_key(identifier)


def read_worksheet(ss: gspread.Spreadsheet, worksheet_name: Optional[str]) -> List[Dict[str, Any]]:
    """
    Read all rows from a worksheet as a list of dicts.

    Args:
        ss: Spreadsheet handle.
        worksheet_name: Name of worksheet (if None, uses first sheet).

    Returns:
        List of dict rows (header-driven).
    """
    ws = ss.worksheet(worksheet_name) if worksheet_name else ss.sheet1
    return ws.get_all_records()


def to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert rows into a DataFrame and canonicalize column names.

    Processing:
        - Renames common variants to canonical names:
            * linkedin_url: ('linkedin', 'linkedin url', 'profile url', 'profile_url', ...)
            * full_name:    ('full name', 'fullname', 'name')
            * company_name: ('company', 'company name', 'employer')
            * title:        ('title', 'job title')
            * description:  ('description', 'summary', 'about')
        - Normalizes linkedin_url with `normalize_linkedin_url`.

    Args:
        rows: List of dict rows.

    Returns:
        Canonicalized pandas DataFrame.
    """
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    rename_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in {"linkedin", "linkedin url", "linkedin_url", "profile url", "profile_url", "profileurl"}:
            rename_map[col] = "linkedin_url"
        elif lc in {"full name", "fullname", "name", "fullname"}:  # already matches 'fullName' via lower()
            rename_map[col] = "full_name"
        elif lc in {"company", "company name", "employer", "companyname"}:
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


def write_dataframe_to_new_worksheet(
    ss: gspread.Spreadsheet, worksheet_name: str, df: pd.DataFrame
) -> None:
    """
    Create (or recreate) a worksheet and write DataFrame contents.

    Processing:
        - Deletes existing worksheet with same name if present.
        - Creates a fresh worksheet sized to fit the DataFrame.
        - Writes headers and all rows.

    Args:
        ss: Spreadsheet to write to.
        worksheet_name: Target worksheet name (will be recreated).
        df: DataFrame to write.
    """
    try:
        try:
            existing = ss.worksheet(worksheet_name)
            ss.del_worksheet(existing)
        except gspread.exceptions.WorksheetNotFound:
            pass

        rows = max(1000, len(df) + 10)
        cols = max(26, len(df.columns))
        ws = ss.add_worksheet(title=worksheet_name, rows=str(rows), cols=str(cols))

        payload = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
        ws.update(payload)
    except Exception as e:
        raise RuntimeError(f"Failed writing worksheet '{worksheet_name}': {e}")


def mark_processed_by_urls(
    ss: gspread.Spreadsheet,
    worksheet_name: Optional[str],
    urls: List[str],
    url_header_candidates: List[str] = None,
    processed_header: str = "processed",
) -> int:
    """
    Mark `processed` = TRUE for rows whose URL matches any in `urls`.

    Args:
        ss: Spreadsheet handle.
        worksheet_name: Worksheet name (None => first sheet).
        urls: Iterable of LinkedIn URLs to mark (must match the sheet's URL column values).
        url_header_candidates: Header names to search for URL column (case-insensitive).
        processed_header: Name of the processed column (created if missing).

    Returns:
        Count of rows updated.
    """
    import gspread

    def _get_ws(s, name):
        return s.worksheet(name) if name else s.sheet1

    def _find_idx(ws, names):
        hdr = ws.row_values(1)
        lower = [h.strip().lower() for h in hdr]
        for n in names:
            nlc = n.strip().lower()
            if nlc in lower:
                return lower.index(nlc) + 1
        return None

    def _ensure_col(ws, header_name):
        hdr = ws.row_values(1)
        lower = [h.strip().lower() for h in hdr]
        name_lc = header_name.strip().lower()
        if name_lc in lower:
            return lower.index(name_lc) + 1
        last_col = len(hdr) + 1
        ws.update_cell(1, last_col, header_name)
        return last_col

    url_header_candidates = url_header_candidates or ["profileUrl", "linkedin_url", "profile url", "profile_url"]
    ws = _get_ws(ss, worksheet_name)
    url_col_idx = _find_idx(ws, url_header_candidates)
    if not url_col_idx:
        raise RuntimeError("URL column not found in worksheet header.")

    processed_col_idx = _ensure_col(ws, processed_header)

    all_vals = ws.get_all_values()
    num_rows = len(all_vals)
    if num_rows <= 1:
        return 0

    url_set = {u.strip() for u in urls if isinstance(u, str) and u.strip()}
    if not url_set:
        return 0

    url_range = ws.range(2, url_col_idx, num_rows, url_col_idx)
    proc_range = ws.range(2, processed_col_idx, num_rows, processed_col_idx)

    updates = 0
    for url_cell, proc_cell in zip(url_range, proc_range):
        url_val = (url_cell.value or "").strip()
        if url_val in url_set and (proc_cell.value or "").strip().upper() != "TRUE":
            proc_cell.value = "TRUE"
            updates += 1

    if updates:
        ws.update_cells(proc_range, value_input_option="USER_ENTERED")

    return updates
