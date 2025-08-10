"""
LLM provider construction and prompt/chain assembly with LangChain.
"""

from typing import Optional

# ✔ robust import for newer + older LC versions
try:
    from langchain_core.output_parsers import JsonOutputParser
except Exception:  # fallback for older installs
    from langchain.output_parsers import JsonOutputParser  # type: ignore

from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import Runnable
from langchain.schema import BaseOutputParser

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .models import Categorization


import os
from dotenv import load_dotenv
load_dotenv()
# ...

def build_llm(provider: str, model: Optional[str] = None, temperature: float = 0.1):
    p = (provider or "").lower()
    if p == "openai":
        model = model or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    elif p in {"google", "gemini"}:
        model = model or "gemini-1.5-pro"
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    else:
        raise ValueError("provider must be 'openai' or 'google'.")

def build_chain(llm) -> Runnable:
    """
    Assemble a prompt → model → JSON parser chain enforcing Categorization schema.
    """
    parser: BaseOutputParser = JsonOutputParser(pydantic_object=Categorization)

    rubric = """
You are a B2B lead categorization assistant.

Output STRICTLY valid JSON with the following keys:
- category: string
- confidence: number (0–1)
- is_researcher: boolean
- is_scholar: boolean  # true if the person is an academic/researcher/scholar
- description: string  # concise description of the person based on LinkedIn profile and activity
- tags: list of 3–7 short tokens
- reasons: string

Classify each person based on their LinkedIn profile + recent activity.

Heuristics:
- "is_researcher" if PhD, publications, 'Research', 'Scholar', 'Professor', or clear academic signals.
- "is_scholar" follows same criteria as is_researcher but also true for anyone with clear scholarly/academic contribution.
- "needs_scholar_lookup" if is_scholar likely True OR unclear but academic signals present.

Return ONLY JSON. No markdown, no code fences.
""".strip()
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

    # Note: parser supplies the format instructions for the schema.
    chain = prompt | llm | parser
    return chain.bind(
        rubric=rubric,
        format_instructions=JsonOutputParser(pydantic_object=Categorization).get_format_instructions(),
    )