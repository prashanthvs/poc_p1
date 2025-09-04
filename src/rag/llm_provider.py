import os
from typing import Literal, Optional
import requests

from langchain_openai import ChatOpenAI

from ..utils.databricks_config import databricks_config


Provider = Literal["together", "openai", "ollama", "databricks"]


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
    - databricks: uses Databricks AI/ML Models serving endpoint
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

    if provider == "databricks":
        # Use Databricks AI/ML Models serving endpoint
        return get_databricks_llm(model, temperature, timeout)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_databricks_llm(model: str, temperature: float, timeout: int):
    """Get LLM from Databricks AI/ML Models serving endpoint."""
    class DatabricksLLM:
        def __init__(self, model: str, temperature: float, timeout: int):
            self.model = model
            self.temperature = temperature
            self.timeout = timeout
            self.endpoint = databricks_config.model_serving_endpoint
            self.access_token = databricks_config.access_token
            self.workspace_url = databricks_config.workspace_url
            
        def invoke(self, messages, **kwargs):
            """Invoke the Databricks model serving endpoint."""
            try:
                # Prepare the request payload
                payload = {
                    "messages": self._format_messages(messages),
                    "temperature": self.temperature,
                    "max_tokens": kwargs.get("max_tokens", 1000)
                }
                
                # Make request to Databricks serving endpoint
                headers = {
                    "Authorization": f"Bearer {self.access_token}",
                    "Content-Type": "application/json"
                }
                
                url = f"https://{self.workspace_url}/serving-endpoints/{self.endpoint}/invocations"
                
                response = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
            except Exception as e:
                print(f"Error calling Databricks model: {e}")
                return "Error: Unable to get response from Databricks model"
        
        def _format_messages(self, messages):
            """Format messages for Databricks API."""
            formatted = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    formatted.append({
                        "role": msg.type if hasattr(msg, 'type') else "user",
                        "content": msg.content
                    })
                else:
                    formatted.append(msg)
            return formatted
        
        def __call__(self, messages, **kwargs):
            return self.invoke(messages, **kwargs)
    
    return DatabricksLLM(model, temperature, timeout)


