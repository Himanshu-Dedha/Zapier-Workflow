# tests/publications_test.py
import os
from dotenv import load_dotenv
load_dotenv()

from app.publications import fetch_publications

def run(name: str, company: str | None = None):
    print(f"[TEST] Querying publications for: {name}  ({company or 'no affiliation'})")
    pubs = fetch_publications(name, company)
    if not pubs:
        print("No results.")
        return
    for i, p in enumerate(pubs, 1):
        print(f"{i}. {p.title}  [{p.year or '—'}]  {p.citations or 0} cites  ({p.source})")
        if p.url:
            print(f"   {p.url}")

if __name__ == "__main__":
    # edit these to a known researcher for a quick win
    run(name="Ankush Tyagi", company="ASU Biodesign Institute")
