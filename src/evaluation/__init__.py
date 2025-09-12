"""
Evaluation package for Maverick RAG application.
Contains metrics calculation, telemetry logging, and performance evaluation tools.
"""

# Correctly import classes from their respective modules
from .metrics import EvaluationMetrics, calculate_precision, calculate_recall, calculate_accuracy
from .telemetry import TelemetryLogger, UserInteraction, SessionTracker, InteractionType, EventType
from .delta_tables import DeltaTableManager

# Define what is available for import from the package
__all__ = [
    "EvaluationMetrics",
    "calculate_precision", 
    "calculate_recall", 
    "calculate_accuracy",
    "TelemetryLogger",
    "UserInteraction",
    "SessionTracker",
    "DeltaTableManager",
    "InteractionType",
    "EventType"
]