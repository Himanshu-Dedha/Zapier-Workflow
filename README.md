# LinkedIn Categorizer Service

A FastAPI service that:
- Reads two Google Sheets (profiles + activities)
- Joins on LinkedIn URL (with normalization)
- Sends each record to an LLM (OpenAI or Google/Gemini) for categorization
- If researcher-like, fetches publications (SerpAPI → Google Scholar; fallback: Semantic Scholar)
- Writes results to a new worksheet

## 1. Prereqs

- Python 3.10+
- A Google Cloud **Service Account** with Sheets and Drive access
- (LLM) OpenAI API key **or** Google AI Studio API key
- (Optional) SerpAPI key (Google Scholar engine)
- (Optional) Semantic Scholar key

## 2. Get credentials

### A) Google Service Account (for Sheets)
1. Go to Google Cloud Console → Create a project (if you don’t have one).
2. Enable APIs:
   - **Google Sheets API**
   - **Google Drive API**
3. Create a **Service Account** → “Keys” → “Add key” → **Create new key** (JSON).
4. Download the JSON key:
   - Either set `GOOGLE_SERVICE_ACCOUNT_FILE` to the absolute file path, **or**
   - Paste the **entire JSON** into `GOOGLE_SERVICE_ACCOUNT_INFO` as a single line.
5. Share each target spreadsheet with the service account’s email (Editor or at least Viewer for reading, Editor for writing).

### B) OpenAI (if using OpenAI provider)
- Create an account at https://platform.openai.com/
- Create an API key and set `OPENAI_API_KEY` in `.env`.

### C) Google Gemini (if using Google provider)
- Go to https://ai.google.dev/ (Google AI Studio), create an API key
- Set `GOOGLE_API_KEY` in `.env`.

### D) SerpAPI (optional, for Google Scholar)
- https://serpapi.com/ → Sign up → copy API key to `SERPAPI_API_KEY`.

### E) Semantic Scholar (optional fallback)
- https://www.semanticscholar.org/product/api → create API key → set `S2_API_KEY`.

## 3. Configure `.env`
Copy `.env.example` to `.env` and fill values. The default sheet references can be overridden per-request.

## 4. Install & run
```bash
pip install -r requirements.txt
# Dev run:
bash run_local.sh
# Or:
uvicorn app.server:app --host 0.0.0.0 --port 8080
