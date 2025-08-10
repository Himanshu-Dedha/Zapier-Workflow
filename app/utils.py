"""
Utility helpers (URL normalization, dict helpers).
"""

import re
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, Optional


def first_nonempty(d: Dict[str, Any], candidates: Iterable[str]) -> Optional[str]:
    """
    Return the first non-empty string from dict `d` among `candidates`.

    Args:
        d: Source dictionary (row).
        candidates: Possible keys to probe.

    Returns:
        First found non-empty string value; otherwise None.
    """
    for c in candidates:
        v = d.get(c)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return None


def normalize_linkedin_url(raw: Optional[str]) -> Optional[str]:
    """
    Normalize LinkedIn URL into a canonical form for joining.

    Processing:
        - Adds 'https://' if missing.
        - Converts netloc to lowercase, strips leading 'www.'.
        - Removes trailing slashes from path.
        - Drops query and fragment parts.
        - Handles inputs like 'in/john-doe' or 'linkedin.com/in/john-doe'.

    Args:
        raw: The raw LinkedIn URL or path-like string.

    Returns:
        Canonicalized LinkedIn URL string, or None if input empty.
    """
    if not raw:
        return None
    u = str(raw).strip()
    if not u:
        return None
    if not u.lower().startswith("http"):
        if u.lower().startswith("linkedin.com"):
            u = "https://" + u
        else:
            u = "https://www.linkedin.com/" + u.lstrip("/")
    try:
        p = urlparse(u)
        netloc = p.netloc.lower().replace("www.", "")
        path = re.sub(r"/+$", "", p.path)
        return f"https://{netloc}{path}"
    except Exception:
        return u.lower()


def safe_get(d: Dict[str, Any], *path, default=None):
    """
    Safely walk nested dict keys.

    Args:
        d: Source dict.
        *path: Key sequence to traverse.
        default: Returned if path not found or None encountered.

    Returns:
        Found value or default.
    """
    cur = d
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p)
    return cur if cur is not None else default
