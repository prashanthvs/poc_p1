# src/utils/config_manager.py
"""
Centralized configuration manager for Maverick RAG application.
"""
import os
import yaml
from typing import Optional, Dict, Any
from dataclasses import dataclass
from databricks_config import DatabricksConfig

@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str
    model: str
    temperature: float
    timeout: int
    api_key: Optional[str] = None
    base_url: Optional[str] = None

@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str
    provider: str

@dataclass
class DataConfig:
    """Data storage and processing configuration."""
    docs_dir: str
    index_dir: str
    use_databricks: bool
    volume_path: Optional[str] = None

@dataclass
class AppConfig:
    """Application configuration."""
    server_name: str
    server_port: int
    title: str

class ConfigManager:
    """Centralized configuration manager for the entire application."""

    def __init__(self, config_path="../src/utils/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._databricks_config: Optional[DatabricksConfig] = None
        self._llm_config: Optional[LLMConfig] = None
        self._embedding_config: Optional[EmbeddingConfig] = None
        self._data_config: Optional[DataConfig] = None
        self._app_config: Optional[AppConfig] = None

    @property
    def databricks(self) -> DatabricksConfig:
        """Get Databricks configuration."""
        if self._databricks_config is None:
            self._databricks_config = self._load_databricks_config()
        return self._databricks_config

    @property
    def llm(self) -> LLMConfig:
        """Get LLM configuration."""
        if self._llm_config is None:
            self._llm_config = self._load_llm_config()
        return self._llm_config

    @property
    def embedding(self) -> EmbeddingConfig:
        """Get embedding configuration."""
        if self._embedding_config is None:
            self._embedding_config = self._load_embedding_config()
        return self._embedding_config

    @property
    def data(self) -> DataConfig:
        """Get data configuration."""
        if self._data_config is None:
            self._data_config = self._load_data_config()
        return self._data_config

    @property
    def app(self) -> AppConfig:
        """Get application configuration."""
        if self._app_config is None:
            self._app_config = self._load_app_config()
        return self._app_config

    def _load_databricks_config(self) -> DatabricksConfig:
        """Load Databricks configuration from YAML and environment variables."""
        db_config = self.config.get("databricks", {})
        return DatabricksConfig(
            workspace_url=os.getenv("DATABRICKS_WORKSPACE_URL", db_config.get("workspace_url")),
            access_token=os.getenv("DATABRICKS_ACCESS_TOKEN", db_config.get("access_token")),
            catalog=os.getenv("DATABRICKS_CATALOG", db_config.get("catalog")),
            schema=os.getenv("DATABRICKS_SCHEMA", db_config.get("schema")),
            volume=os.getenv("DATABRICKS_VOLUME", db_config.get("volume")),
            vector_search_endpoint=os.getenv("DATABRICKS_VECTOR_SEARCH_ENDPOINT", db_config.get("vector_search_endpoint")),
            model_serving_endpoint=os.getenv("DATABRICKS_MODEL_SERVING_ENDPOINT", db_config.get("model_serving_endpoint"))
        )

    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration from YAML and environment variables."""
        llm_config = self.config.get("llm", {})
        provider = os.getenv("LLM_PROVIDER", llm_config.get("provider"))
        provider_settings = llm_config.get("providers", {}).get(provider, {})

        api_key_env_map = {
            "together": "TOGETHER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "ollama": "OLLAMA_API_KEY",
            "databricks": "DATABRICKS_ACCESS_TOKEN"
        }
        api_key = os.getenv(api_key_env_map.get(provider))

        return LLMConfig(
            provider=provider,
            model=os.getenv("LLAMA_MODEL", llm_config.get("model")),
            temperature=float(os.getenv("LLM_TEMPERATURE", llm_config.get("temperature"))),
            timeout=int(os.getenv("LLM_TIMEOUT", llm_config.get("timeout"))),
            api_key=api_key,
            base_url=provider_settings.get("base_url")
        )

    def _load_embedding_config(self) -> EmbeddingConfig:
        """Load embedding configuration from YAML."""
        embed_config = self.config.get("embedding", {})
        return EmbeddingConfig(
            model_name=embed_config.get("model_name"),
            provider=embed_config.get("provider")
        )

    def _load_data_config(self) -> DataConfig:
        """Load data configuration from YAML."""
        data_config = self.config.get("data", {})
        use_databricks = os.getenv("USE_DATABRICKS", str(data_config.get("use_databricks"))).lower() == "true"
        return DataConfig(
            docs_dir=data_config.get("docs_dir"),
            index_dir=data_config.get("index_dir"),
            use_databricks=use_databricks,
            volume_path=self.databricks.volume_path if use_databricks else None
        )

    def _load_app_config(self) -> AppConfig:
        """Load application configuration from YAML."""
        app_config = self.config.get("app", {})
        return AppConfig(
            server_name=app_config.get("server_name"),
            server_port=int(app_config.get("server_port")),
            title=app_config.get("title")
        )

# Global configuration manager instance
config_manager = ConfigManager()