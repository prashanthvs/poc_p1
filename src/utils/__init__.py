"""
Utils package for Maverick RAG application.
Contains configuration management and utility functions.
"""

from .config_manager import config_manager
from .databricks_config import DatabricksConfig

databricks_config = config_manager.databricks()

__all__ = [
    "config_manager",
    "DatabricksConfig", 
    "databricks_config"
]
