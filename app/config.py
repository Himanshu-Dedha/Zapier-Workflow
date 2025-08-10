"""
Configuration loader for environment variables.

Reads `.env` using python-dotenv so you can keep secrets out of code.
Provides a centralized config object for the rest of the app.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Application configuration.

    Attributes:
        SERVICE_API_KEY: Shared secret for protecting the FastAPI endpoint.
        HOST: Host binding for uvicorn (default '0.0.0.0').
        PORT: Port for uvicorn (default 8080).

        GOOGLE_SERVICE_ACCOUNT_FILE: Absolute path to service account JSON key file.
        GOOGLE_SERVICE_ACCOUNT_INFO: Inline JSON (single-line) content of the service account key.

        PROFILES_SHEET_URL_OR_ID: Default profiles sheet URL or ID.
        PROFILES_WORKSHEET: Default profiles worksheet name.
        ACTIVITY_SHEET_URL_OR_ID: Default activities sheet URL or ID.
        ACTIVITY_WORKSHEET: Default activities worksheet name.
        OUTPUT_WORKSHEET: Default output worksheet name ('Categorized').

        LLM_PROVIDER: 'openai' or 'google'.
        LLM_MODEL: Model name (e.g., 'gpt-4o-mini' or 'gemini-1.5-pro').
        LLM_TEMPERATURE: Sampling temperature.

        OPENAI_API_KEY: OpenAI key (if provider=openai).
        GOOGLE_API_KEY: Google AI Studio key (if provider=google).

        SERPAPI_API_KEY: SerpAPI key for Google Scholar (optional).
        S2_API_KEY: Semantic Scholar key (optional).
    """

    # Server
    SERVICE_API_KEY: str = "change-this"
    HOST: str = "0.0.0.0"
    PORT: int = 8080

    # Google SA (choose one method)
    GOOGLE_SERVICE_ACCOUNT_FILE: Optional[str] = None
    GOOGLE_SERVICE_ACCOUNT_INFO: Optional[str] = None

    # Default Sheets
    PROFILES_SHEET_URL_OR_ID: Optional[str] = None
    PROFILES_WORKSHEET: Optional[str] = None
    ACTIVITY_SHEET_URL_OR_ID: Optional[str] = None
    ACTIVITY_WORKSHEET: Optional[str] = None
    OUTPUT_WORKSHEET: str = "Categorized"

    # LLM defaults
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1

    # Provider keys
    OPENAI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None

    # Publications keys
    SERPAPI_API_KEY: Optional[str] = None
    S2_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
