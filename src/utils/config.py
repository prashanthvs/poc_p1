# src/utils/config.py
"""
Centralized, single-file configuration manager for the Maverick RAG application.
"""
import os, sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from databricks.sdk import WorkspaceClient

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
                    "workspace_url and access_token must be set in configuration.yaml for Databricks integration"
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
                workspace_url=db_config.get("workspace_url"),
                access_token=db_config.get("access_token"),
                catalog=db_config.get("catalog"),
                schema=db_config.get("schema"),
                volume=db_config.get("volume"),
                vector_search_endpoint=db_config.get("vector_search_endpoint"),
                model_serving_endpoint=db_config.get("model_serving_endpoint")
            )
        return self._databricks

    @property
    def llm(self) -> LLMConfig:
        if self._llm is None:
            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider")
            provider_settings = llm_config.get("providers", {}).get(provider, {})

            self._llm = LLMConfig(
                provider=provider,
                model=llm_config.get("model"),
                temperature=float(llm_config.get("temperature", 0.2)),
                timeout=int(llm_config.get("timeout", 60)),
                api_key=provider_settings.get("api_key"),
                base_url=provider_settings.get("base_url")
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
            use_db = str(data_config.get("use_databricks", "false")).lower() == "true"
            self._data = DataConfig(
                docs_dir=str(PROJECT_ROOT / data_config.get("docs_dir")),
                index_dir=str(PROJECT_ROOT / data_config.get("index_dir")),
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
        if self.llm.provider in ["together", "openai"] and not self.llm.api_key:
            errors.append(f"LLM_PROVIDER is '{self.llm.provider}' but its api_key is not set in configuration.yaml.")

        # Validate Databricks configuration if enabled
        if self.data.use_databricks:
            if not self.databricks.workspace_url:
                errors.append("USE_DATABRICKS is true but databricks.workspace_url is not set.")
            if not self.databricks.access_token:
                errors.append("USE_DATABRICKS is true but databricks.access_token is not set.")
            if not self.databricks.vector_search_endpoint:
                errors.append("USE_DATABRICKS is true but databricks.vector_search_endpoint is not set.")
            if self.llm.provider == "databricks" and not self.databricks.model_serving_endpoint:
                 errors.append("LLM_PROVIDER is 'databricks' but databricks.model_serving_endpoint is not set.")
        
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
config_manager = ConfigManager()