# src/utils/databricks_config.py
"""
Databricks configuration module for Unity Catalog and AI/ML Models integration.
"""
from typing import Optional, Dict, Any
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeInfo, VolumeType

# ⬇️ REPLACEMENT: use MLflow Deployments for serving endpoint info
from mlflow.deployments import get_deploy_client
import os


class DatabricksConfig:
    """Configuration class for Databricks Unity Catalog and AI/ML Models."""

    def __init__(
        self,
        workspace_url: str,
        access_token: str,
        catalog: str,
        schema: str,
        volume: str,
        vector_search_endpoint: Optional[str],
        model_serving_endpoint: Optional[str],
    ):
        self.workspace_url = workspace_url
        self.access_token = access_token
        self.catalog_name = catalog
        self.schema_name = schema
        self.volume_name = volume
        self.vector_search_endpoint = vector_search_endpoint
        self.model_serving_endpoint = model_serving_endpoint

        # Initialize workspace client
        self._client: Optional[WorkspaceClient] = None

    @property
    def client(self) -> WorkspaceClient:
        """Get or create Databricks workspace client."""
        if self._client is None:
            if not self.workspace_url or not self.access_token:
                raise ValueError(
                    "DATABRICKS_WORKSPACE_URL and DATABRICKS_ACCESS_TOKEN must be set"
                )
            self._client = WorkspaceClient(
                host=self.workspace_url,
                token=self.access_token
            )
        return self._client

    @property
    def volume_path(self) -> str:
        """Get the full Unity Catalog volume path."""
        return f"/Volumes/{self.catalog_name}/{self.schema_name}/{self.volume_name}"

    @property
    def vector_search_index_name(self) -> str:
        """Get the vector search index name."""
        return f"{self.catalog_name}.{self.schema_name}.rag_embeddings"

    def get_volume_info(self) -> Optional[VolumeInfo]:
        """Get information about the Unity Catalog volume."""
        try:
            return self.client.volumes.get(
                f"{self.catalog_name}.{self.schema_name}.{self.volume_name}"
            )
        except Exception as e:
            print(f"Warning: Could not get volume info: {e}")
            return None

    def create_volume_if_not_exists(self) -> VolumeInfo:
        """Create Unity Catalog volume if it doesn't exist."""
        try:
            # Try to get existing volume
            return self.client.volumes.get(
                f"{self.catalog_name}.{self.schema_name}.{self.volume_name}"
            )
        except Exception:
            # Create new volume
            return self.client.volumes.create(
                catalog_name=self.catalog_name,
                schema_name=self.schema_name,
                name=self.volume_name,
                volume_type=VolumeType.MANAGED
            )

    def get_model_serving_endpoint(self) -> Optional[Dict[str, Any]]:
        """
        Get the model serving endpoint configuration via MLflow Deployments.

        Uses MLflow's Databricks deployment client, which reads credentials from
        DATABRICKS_HOST / DATABRICKS_TOKEN and provides a dict-like endpoint object.
        """
        if not self.model_serving_endpoint:
            return None

        # Preserve existing env values if any
        prev_host = os.environ.get("DATABRICKS_HOST")
        prev_token = os.environ.get("DATABRICKS_TOKEN")

        try:
            # Ensure MLflow uses this instance's credentials
            os.environ["DATABRICKS_HOST"] = self.workspace_url
            os.environ["DATABRICKS_TOKEN"] = self.access_token

            client = get_deploy_client("databricks")  # MLflow Deployments client
            # get_endpoint(endpoint=...) returns a DatabricksEndpoint (dict-like)
            endpoint = client.get_endpoint(endpoint=self.model_serving_endpoint)
            return dict(endpoint)  # normalize to plain dict
        except Exception as e:
            print(f"Warning: Could not get model serving endpoint: {e}")
            return None
        finally:
            # Restore env
            if prev_host is None:
                os.environ.pop("DATABRICKS_HOST", None)
            else:
                os.environ["DATABRICKS_HOST"] = prev_host
            if prev_token is None:
                os.environ.pop("DATABRICKS_TOKEN", None)
            else:
                os.environ["DATABRICKS_TOKEN"] = prev_token

    def get_vector_search_endpoint(self) -> Optional[Dict[str, Any]]:
        """Get the vector search endpoint configuration."""
        if not self.vector_search_endpoint:
            return None
        try:
            # Note: This would need to be implemented based on the actual vector search API
            # For now, return a placeholder
            return {"endpoint_name": self.vector_search_endpoint}
        except Exception as e:
            print(f"Warning: Could not get vector search endpoint: {e}")
            return None
        
databricks_config = DatabricksConfig()
