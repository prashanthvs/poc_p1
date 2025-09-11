import os
from typing import Literal, Optional
import requests

from langchain_openai import ChatOpenAI
from src.utils.config import config_manager


Provider = Literal["together", "openai", "ollama", "databricks"]


def get_llm(
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = None,
    timeout: int = None,
):
    """Return a LangChain ChatModel configured for the chosen provider.

    Supported providers:
    - together: uses Together AI for Meta Llama models
    - openai: any OpenAI-compatible model
    - ollama: local Ollama server with llama models
    - databricks: uses Databricks AI/ML Models serving endpoint
    """

    # Use configuration from config_manager as the primary source
    llm_config = config_manager.llm
    
    provider = provider or llm_config.provider
    model = model or llm_config.model
    api_key = api_key if api_key is not None else llm_config.api_key
    temperature = temperature if temperature is not None else llm_config.temperature
    timeout = timeout or llm_config.timeout

    if provider == "together":
        # Configure OpenAI-compatible client with Together base URL
        # Requires TOGETHER_API_KEY
        return ChatOpenAI(
            model=model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            temperature=temperature,
            timeout=timeout,
        )

    if provider == "openai":
        # Native OpenAI
        return ChatOpenAI(
            model=model,
            api_key=llm_config.api_key,
            temperature=temperature,
            timeout=timeout,
        )

    if provider == "ollama":
        # Use OpenAI-compatible wrapper pointed at Ollama
        return ChatOpenAI(
            model=model,
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
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
            self.endpoint = config_manager.databricks.model_serving_endpoint
            self.access_token = config_manager.databricks.access_token
            self.workspace_url = config_manager.databricks.workspace_url
            
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