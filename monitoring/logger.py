"""
TORCO Pest Control Recommendation System - Logging Module

This module provides clean prediction logging functionality for the monitoring system.
It handles structured logging of predictions with proper formatting and error handling.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class PredictionLogger:
    """
    Structured logger for recommender system predictions.

    This class provides clean logging of prediction events with proper formatting,
    error handling, and integration with the drift detection system.
    """

    def __init__(self, log_dir: str = "monitoring/logs"):
        """
        Initialize the prediction logger.

        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up Python logging
        self.logger = logging.getLogger('torco_recommender')
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # File handler for detailed logs
        log_file = self.log_dir / f"predictions_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)

        # JSON log file for structured data
        self.json_log_file = self.log_dir / "predictions.jsonl"

    def log_prediction(self, customer_id: str, service_id: str, score: float,
                      urgency: int, context: Optional[Dict[str, Any]] = None,
                      timestamp: Optional[str] = None) -> None:
        """
        Log a prediction event with structured data.

        Args:
            customer_id: Unique customer identifier
            service_id: Recommended service identifier
            score: Recommendation score (0-1)
            urgency: Urgency score (1-10)
            context: Additional context (season, property_type, etc.)
            timestamp: ISO format timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        # Create structured log entry
        log_entry = {
            "timestamp": timestamp,
            "customer_id": customer_id,
            "service_id": service_id,
            "score": float(score),
            "urgency": int(urgency),
            "context": context or {}
        }

        try:
            # Log to Python logger
            self.logger.info(
                f"Prediction: customer={customer_id}, service={service_id}, "
                f"score={score:.3f}, urgency={urgency}"
            )

            # Log to JSON file
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            self.logger.error(f"Failed to log prediction: {e}")
            # Don't raise exception to avoid breaking the main flow

    def log_error(self, error_type: str, message: str,
                 context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error event.

        Args:
            error_type: Type of error (e.g., 'prediction_error', 'drift_alert')
            message: Error message
            context: Additional context information
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "message": message,
            "context": context or {}
        }

        try:
            self.logger.error(f"{error_type}: {message}")

            # Log error to JSON file
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            print(f"Failed to log error: {e}")  # Fallback logging

    def get_recent_logs(self, hours: int = 24) -> list:
        """
        Get recent prediction logs.

        Args:
            hours: Number of hours to look back

        Returns:
            List of recent log entries
        """
        try:
            logs = []
            cutoff_time = datetime.now().timestamp() - (hours * 3600)

            if self.json_log_file.exists():
                with open(self.json_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                            if entry_time >= cutoff_time:
                                logs.append(entry)
                        except (json.JSONDecodeError, KeyError):
                            continue

            return logs

        except Exception as e:
            self.logger.error(f"Failed to read recent logs: {e}")
            return []


# Global logger instance
_logger = None


def get_logger() -> PredictionLogger:
    """Get the global prediction logger instance."""
    global _logger
    if _logger is None:
        _logger = PredictionLogger()
    return _logger


def log_prediction(customer_id: str, service_id: str, score: float,
                  urgency: int, context: Optional[Dict[str, Any]] = None,
                  timestamp: Optional[str] = None) -> None:
    """
    Log a prediction using the global logger.

    Args:
        customer_id: Unique customer identifier
        service_id: Recommended service identifier
        score: Recommendation score (0-1)
        urgency: Urgency score (1-10)
        context: Additional context information
        timestamp: ISO format timestamp (auto-generated if None)
    """
    logger = get_logger()
    logger.log_prediction(customer_id, service_id, score, urgency, context, timestamp)


def log_error(error_type: str, message: str,
             context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error using the global logger.

    Args:
        error_type: Type of error
        message: Error message
        context: Additional context information
    """
    logger = get_logger()
    logger.log_error(error_type, message, context)