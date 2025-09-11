# src/utils/config.py
"""
Centralized, single-file configuration manager for the Maverick RAG application.
"""
import os, sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client

# This makes path resolution independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# --- Configuration Data Classes ---

@dataclass
class DatabricksConfig:
    """Configuration class for Databricks integration."""
    workspace_url: Optional[str] = None
    access_token: Optional[str] = None
    catalog: str = "main"
    schema: str = "default"
    volume: str = "source_data"
    vector_search_endpoint: Optional[str] = None
    model_serving_endpoint: Optional[str] = None
    _client: Optional[WorkspaceClient] = field(init=False, repr=False, default=None)

    @property
    def client(self) -> WorkspaceClient:
        """Get or create Databricks workspace client."""
        if self._client is None:
            if not self.workspace_url or not self.access_token:
                raise ValueError(
                    "DATABRICKS_WORKSPACE_URL and DATABRICKS_ACCESS_TOKEN must be set for Databricks integration"
                )
            self._client = WorkspaceClient(host=self.workspace_url, token=self.access_token)
        return self._client

    @property
    def volume_path(self) -> str:
        """Get the full Unity Catalog volume path."""
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

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

# --- The Main ConfigManager Class ---

class ConfigManager:
    """Manages all application configurations from a single source."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # Construct path relative to this file's location for robustness.
            # This ensures the config file is found whether running from the root or from /notebooks.
            base_dir = Path(__file__).resolve().parent
            config_path = base_dir / "configuration.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self._databricks: Optional[DatabricksConfig] = None
        self._llm: Optional[LLMConfig] = None
        self._embedding: Optional[EmbeddingConfig] = None
        self._data: Optional[DataConfig] = None
        self._app: Optional[AppConfig] = None

    @property
    def databricks(self) -> DatabricksConfig:
        if self._databricks is None:
            db_config = self.config.get("databricks", {})
            self._databricks = DatabricksConfig(
                workspace_url=os.getenv("DATABRICKS_WORKSPACE_URL", db_config.get("workspace_url")),
                access_token=os.getenv("DATABRICKS_ACCESS_TOKEN", db_config.get("access_token")),
                catalog=os.getenv("DATABRICKS_CATALOG", db_config.get("catalog")),
                schema=os.getenv("DATABRICKS_SCHEMA", db_config.get("schema")),
                volume=os.getenv("DATABRICKS_VOLUME", db_config.get("volume")),
                vector_search_endpoint=os.getenv("DATABRICKS_VECTOR_SEARCH_ENDPOINT", db_config.get("vector_search_endpoint")),
                model_serving_endpoint=os.getenv("DATABRICKS_MODEL_SERVING_ENDPOINT", db_config.get("model_serving_endpoint"))
            )
        return self._databricks

    @property
    def llm(self) -> LLMConfig:
        if self._llm is None:
            llm_config = self.config.get("llm", {})
            provider = os.getenv("LLM_PROVIDER", llm_config.get("provider"))
            
            api_key_env_map = {
                "together": "TOGETHER_API_KEY", "openai": "OPENAI_API_KEY",
                "ollama": "OLLAMA_API_KEY", "databricks": "DATABRICKS_ACCESS_TOKEN"
            }
            api_key = os.getenv(api_key_env_map.get(provider))
            
            self._llm = LLMConfig(
                provider=provider,
                model=os.getenv("LLAMA_MODEL", llm_config.get("model")),
                temperature=float(os.getenv("LLM_TEMPERATURE", llm_config.get("temperature", 0.2))),
                timeout=int(os.getenv("LLM_TIMEOUT", llm_config.get("timeout", 60))),
                api_key=api_key,
                base_url=llm_config.get("providers", {}).get(provider, {}).get("base_url")
            )
        return self._llm

    @property
    def embedding(self) -> EmbeddingConfig:
        if self._embedding is None:
            embed_config = self.config.get("embedding", {})
            self._embedding = EmbeddingConfig(
                model_name=embed_config.get("model_name"),
                provider=embed_config.get("provider")
            )
        return self._embedding

    @property
    def data(self) -> DataConfig:
        if self._data is None:
            data_config = self.config.get("data", {})
            use_db = str(os.getenv("USE_DATABRICKS", data_config.get("use_databricks", "false"))).lower() == "true"
            self._data = DataConfig(
                docs_dir=data_config.get("docs_dir"),
                index_dir=data_config.get("index_dir"),
                use_databricks=use_db,
                volume_path=self.databricks.volume_path if use_db else None
            )
        return self._data

    @property
    def app(self) -> AppConfig:
        if self._app is None:
            app_config = self.config.get("app", {})
            self._app = AppConfig(
                server_name=app_config.get("server_name"),
                server_port=int(app_config.get("server_port")),
                title=app_config.get("title")
            )
        return self._app

    def get_config_summary(self) -> Dict[str, Any]:
        """Return a dictionary summary of the current configuration."""
        summary = {
            "app": asdict(self.app),
            "llm": asdict(self.llm),
            "embedding": asdict(self.embedding),
            "data": asdict(self.data),
            "databricks": asdict(self.databricks)
        }
        # Censor sensitive keys
        if summary["llm"].get("api_key"):
            summary["llm"]["api_key"] = "********"
        if summary["databricks"].get("access_token"):
            summary["databricks"]["access_token"] = "********"
        return summary

    def validate_config(self) -> Dict[str, Any]:
        """Validate the configuration and return status, errors, and warnings."""
        errors: List[str] = []
        warnings: List[str] = []

        # Validate LLM provider API keys
        if self.llm.provider == "together" and not self.llm.api_key:
            errors.append("LLM_PROVIDER is 'together' but TOGETHER_API_KEY is not set.")
        elif self.llm.provider == "openai" and not self.llm.api_key:
            errors.append("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")

        # Validate Databricks configuration if enabled
        if self.data.use_databricks:
            if not self.databricks.workspace_url:
                errors.append("USE_DATABRICKS is true but DATABRICKS_WORKSPACE_URL is not set.")
            if not self.databricks.access_token:
                errors.append("USE_DATABRICKS is true but DATABRICKS_ACCESS_TOKEN is not set.")
            if not self.databricks.vector_search_endpoint:
                errors.append("USE_DATABRICKS is true but DATABRICKS_VECTOR_SEARCH_ENDPOINT is not set.")
            if self.llm.provider == "databricks" and not self.databricks.model_serving_endpoint:
                 errors.append("LLM_PROVIDER is 'databricks' but DATABRICKS_MODEL_SERVING_ENDPOINT is not set.")
        
        # Validate local data paths if Databricks is not used
        else:
            if not os.path.exists(self.data.docs_dir):
                warnings.append(f"Local docs_dir '{self.data.docs_dir}' does not exist.")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

# --- Create a Single, Global Instance for the Entire Application ---
# This instance will be imported by other modules.
config_manager = ConfigManager()