"""
Centralized configuration manager for Maverick RAG application.
"""
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .databricks_config import DatabricksConfig


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
    provider: str = "huggingface"


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
    
    def __init__(self):
        self._databricks_config: Optional[DatabricksConfig] = None
        self._llm_config: Optional[LLMConfig] = None
        self._embedding_config: Optional[EmbeddingConfig] = None
        self._data_config: Optional[DataConfig] = None
        self._app_config: Optional[AppConfig] = None
    
    @property
    def databricks(self) -> DatabricksConfig:
        """Get Databricks configuration."""
        if self._databricks_config is None:
            self._databricks_config = DatabricksConfig()
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
    
    def _load_llm_config(self) -> LLMConfig:
        """Load LLM configuration from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "together").lower()
        model = os.getenv("LLAMA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        timeout = int(os.getenv("LLM_TIMEOUT", "60"))
        
        # Get API key based on provider
        api_key = None
        base_url = None
        
        if provider == "together":
            api_key = os.getenv("TOGETHER_API_KEY")
            base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "ollama":
            api_key = os.getenv("OLLAMA_API_KEY", "ollama")
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        elif provider == "databricks":
            api_key = os.getenv("DATABRICKS_ACCESS_TOKEN")
            base_url = os.getenv("DATABRICKS_WORKSPACE_URL")
        
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            timeout=timeout,
            api_key=api_key,
            base_url=base_url
        )
    
    def _load_embedding_config(self) -> EmbeddingConfig:
        """Load embedding configuration from environment variables."""
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
        
        return EmbeddingConfig(
            model_name=model_name,
            provider=provider
        )
    
    def _load_data_config(self) -> DataConfig:
        """Load data configuration from environment variables."""
        docs_dir = os.getenv("DOCS_DIR", "data/docs")
        index_dir = os.getenv("INDEX_DIR", "data/index/faiss")
        use_databricks = os.getenv("USE_DATABRICKS", "false").lower() == "true"
        volume_path = os.getenv("DATABRICKS_VOLUME_PATH")
        
        return DataConfig(
            docs_dir=docs_dir,
            index_dir=index_dir,
            use_databricks=use_databricks,
            volume_path=volume_path
        )
    
    def _load_app_config(self) -> AppConfig:
        """Load application configuration from environment variables."""
        server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
        server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
        title = os.getenv("APP_TITLE", "Maverick RAG")
        
        return AppConfig(
            server_name=server_name,
            server_port=server_port,
            title=title
        )
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results."""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validate LLM configuration
        if not self.llm.api_key and self.llm.provider != "databricks":
            validation_results["errors"].append(f"API key required for {self.llm.provider} provider")
            validation_results["valid"] = False
        
        # Validate Databricks configuration if enabled
        if self.data.use_databricks:
            if not self.databricks.workspace_url:
                validation_results["errors"].append("DATABRICKS_WORKSPACE_URL is required")
                validation_results["valid"] = False
            
            if not self.databricks.access_token:
                validation_results["errors"].append("DATABRICKS_ACCESS_TOKEN is required")
                validation_results["valid"] = False
            
            if not self.databricks.vector_search_endpoint:
                validation_results["warnings"].append("DATABRICKS_VECTOR_SEARCH_ENDPOINT not set")
            
            if not self.databricks.model_serving_endpoint:
                validation_results["warnings"].append("DATABRICKS_MODEL_SERVING_ENDPOINT not set")
        
        # Validate data directories for local mode
        if not self.data.use_databricks:
            if not os.path.exists(self.data.docs_dir):
                validation_results["warnings"].append(f"Documents directory not found: {self.data.docs_dir}")
        
        return validation_results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "has_api_key": bool(self.llm.api_key)
            },
            "embedding": {
                "model": self.embedding.model_name,
                "provider": self.embedding.provider
            },
            "data": {
                "docs_dir": self.data.docs_dir,
                "index_dir": self.data.index_dir,
                "use_databricks": self.data.use_databricks,
                "volume_path": self.data.volume_path
            },
            "app": {
                "server_name": self.app.server_name,
                "server_port": self.app.server_port,
                "title": self.app.title
            },
            "databricks": {
                "workspace_url": self.databricks.workspace_url,
                "catalog": self.databricks.catalog_name,
                "schema": self.databricks.schema_name,
                "volume": self.databricks.volume_name,
                "has_vector_search": bool(self.databricks.vector_search_endpoint),
                "has_model_serving": bool(self.databricks.model_serving_endpoint)
            }
        }


# Global configuration manager instance
config_manager = ConfigManager()
