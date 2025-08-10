# test.py
import os
from dotenv import load_dotenv

# Load env vars from .env in project root
load_dotenv()

from app.llm import build_llm

def main():
    # Read provider/model from env (fallbacks in case missing)
    provider = os.getenv("LLM_PROVIDER", "openai")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini" if provider == "openai" else "gemini-1.5-pro")

    print(f"[TEST] Using provider: {provider}, model: {model}")

    # Build LLM
    llm = build_llm(provider=provider, model=model, temperature=0.2)

    # Sample prompt
    prompt = "Explain in two sentences why the sky is blue."

    print(f"[TEST] Sending prompt: {prompt}")
    try:
        response = llm.invoke(prompt)
        # For LangChain chat models, .invoke() returns a Message
        if hasattr(response, "content"):
            print("\n[TEST] Model Response:")
            print(response.content)
        else:
            print("\n[TEST] Raw Response:")
            print(response)
    except Exception as e:
        print(f"[ERROR] Failed to query LLM: {e}")

if __name__ == "__main__":
    main()
