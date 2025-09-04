import os
from typing import Literal, Optional

from langchain_openai import ChatOpenAI


Provider = Literal["together", "openai", "ollama"]


def get_llm(
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: int = 60,
):
    """Return a LangChain ChatModel configured for the chosen provider.

    Supported providers:
    - together: uses Together AI for Meta Llama models (e.g., "meta-llama/Meta-Llama-3.1-8B-Instruct")
    - openai: any OpenAI-compatible model
    - ollama: local Ollama server with llama models (e.g., "llama3.1")
    """

    provider = provider or os.getenv("LLM_PROVIDER", "together").lower()
    # Default to a Llama-family instruct model. User can override via LLAMA_MODEL.
    model = model or os.getenv(
        "LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    if provider == "together":
        # Configure OpenAI-compatible client with Together base URL
        # Requires TOGETHER_API_KEY
        base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )

    if provider == "openai":
        # Native OpenAI
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=temperature,
            timeout=timeout,
        )

    if provider == "ollama":
        # Use OpenAI-compatible wrapper pointed at Ollama
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        return ChatOpenAI(
            model=model,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


