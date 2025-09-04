"""
Evaluation package for Maverick RAG application.
Contains metrics calculation, telemetry logging, and performance evaluation tools.
"""

from .metrics import EvaluationMetrics, calculate_precision, calculate_recall, calculate_accuracy
from .telemetry import TelemetryLogger, UserInteraction, SessionTracker
from .delta_tables import DeltaTableManager

__all__ = [
    "EvaluationMetrics",
    "calculate_precision", 
    "calculate_recall", 
    "calculate_accuracy",
    "TelemetryLogger",
    "UserInteraction",
    "SessionTracker",
    "DeltaTableManager"
]
