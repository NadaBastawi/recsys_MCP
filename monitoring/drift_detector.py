"""
TORCO Pest Control Recommendation System - Monitoring Module

This module implements data drift detection for production monitoring of the
recommender system. It tracks prediction logs and detects drift in customer
features and service patterns.

Key Features:
- Logs every prediction to a local JSON file
- Detects feature drift using KS test (numerical) and PSI (categorical)
- Provides drift reports with status levels (OK/WARNING/ALERT)
- Thresholds: WARNING at PSI > 0.1, ALERT at PSI > 0.25
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """
    Data drift detection system for monitoring recommender system predictions.

    This class provides methods to log predictions and detect drift in customer
    features and service patterns using statistical tests.
    """

    def __init__(self, log_file: str = "monitoring/prediction_logs.json"):
        """
        Initialize the drift detector.

        Args:
            log_file: Path to the JSON file for storing prediction logs
        """
        self.log_file = log_file
        self._ensure_log_file_exists()

    def _ensure_log_file_exists(self) -> None:
        """Create the log file if it doesn't exist."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def log_prediction(self, customer_id: str, service_id: str, score: float,
                      urgency: int, timestamp: str = None) -> None:
        """
        Log a prediction to the monitoring file.

        Args:
            customer_id: Unique customer identifier
            service_id: Recommended service identifier
            score: Recommendation score (0-1)
            urgency: Urgency score (1-10)
            timestamp: ISO format timestamp (auto-generated if None)
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        prediction = {
            "customer_id": customer_id,
            "service_id": service_id,
            "score": float(score),
            "urgency": int(urgency),
            "timestamp": timestamp
        }

        # Load existing logs
        with open(self.log_file, 'r') as f:
            logs = json.load(f)

        # Append new prediction
        logs.append(prediction)

        # Save back to file
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def _calculate_psi(self, reference: pd.Series, production: pd.Series,
                      bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for categorical/numerical drift.

        Args:
            reference: Reference distribution
            production: Production distribution
            bins: Number of bins for numerical data

        Returns:
            PSI score (lower is better, < 0.1 is OK)
        """
        # Handle numerical data by binning
        if pd.api.types.is_numeric_dtype(reference):
            # Create bins from reference data
            bins_edges = pd.qcut(reference, q=bins, duplicates='drop').cat.categories
            ref_counts = pd.cut(reference, bins=bins_edges).value_counts().sort_index()
            prod_counts = pd.cut(production, bins=bins_edges).value_counts().sort_index()
        else:
            # Categorical data
            ref_counts = reference.value_counts()
            prod_counts = production.value_counts()

        # Calculate percentages
        ref_pct = ref_counts / ref_counts.sum()
        prod_pct = prod_counts / prod_counts.sum()

        # Handle missing categories in production
        all_categories = set(ref_pct.index) | set(prod_pct.index)
        ref_pct = ref_pct.reindex(all_categories, fill_value=0.0001)  # Small epsilon
        prod_pct = prod_pct.reindex(all_categories, fill_value=0.0001)

        # Calculate PSI
        psi = ((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)).sum()

        return float(psi)

    def detect_feature_drift(self, reference_df: pd.DataFrame,
                           production_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift in customer features between reference and production data.

        Args:
            reference_df: Reference dataset (historical)
            production_df: Production dataset (recent predictions)

        Returns:
            Dictionary with drift scores and status for each feature
        """
        drift_report = {}

        # Define features to monitor
        numerical_features = ['building_age_years', 'urgency_score']
        categorical_features = ['property_type', 'season']

        # Check numerical features with KS test
        for feature in numerical_features:
            if feature in reference_df.columns and feature in production_df.columns:
                ref_values = reference_df[feature].dropna()
                prod_values = production_df[feature].dropna()

                if len(ref_values) > 0 and len(prod_values) > 0:
                    # KS test
                    ks_stat, p_value = ks_2samp(ref_values, prod_values)

                    # PSI for additional context
                    psi_score = self._calculate_psi(ref_values, prod_values)

                    drift_report[feature] = {
                        "drift_score": float(ks_stat),
                        "p_value": float(p_value),
                        "psi_score": psi_score,
                        "test_type": "ks_test",
                        "status": "OK" if p_value > 0.05 else ("WARNING" if p_value > 0.01 else "ALERT")
                    }

        # Check categorical features with PSI
        for feature in categorical_features:
            if feature in reference_df.columns and feature in production_df.columns:
                ref_values = reference_df[feature].dropna()
                prod_values = production_df[feature].dropna()

                if len(ref_values) > 0 and len(prod_values) > 0:
                    psi_score = self._calculate_psi(ref_values, prod_values)

                    status = "OK"
                    if psi_score > 0.25:
                        status = "ALERT"
                    elif psi_score > 0.1:
                        status = "WARNING"

                    drift_report[feature] = {
                        "drift_score": psi_score,
                        "test_type": "psi",
                        "status": status
                    }

        return drift_report

    def get_drift_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report from logged predictions.

        Returns:
            Dictionary containing drift analysis results
        """
        # Load prediction logs
        with open(self.log_file, 'r') as f:
            logs = json.load(f)

        if not logs:
            return {"error": "No prediction logs available for drift analysis"}

        # Convert to DataFrame
        df = pd.DataFrame(logs)

        # Create reference dataset (first 70% of logs)
        split_idx = int(len(df) * 0.7)
        reference_df = df.iloc[:split_idx].copy()
        production_df = df.iloc[split_idx:].copy()

        # Add customer features to production data (mock for demo)
        # In real implementation, this would join with actual customer data
        np.random.seed(42)  # For reproducible mock data
        production_df['building_age_years'] = np.random.normal(25, 10, len(production_df))
        production_df['property_type'] = np.random.choice(
            ['residential_house', 'residential_apartment', 'commercial_restaurant',
             'commercial_office', 'commercial_warehouse'],
            len(production_df)
        )
        production_df['season'] = np.random.choice(['winter', 'spring', 'summer', 'fall'], len(production_df))

        # Detect drift
        drift_results = self.detect_feature_drift(reference_df, production_df)

        # Overall status
        statuses = [result['status'] for result in drift_results.values()]
        overall_status = "OK"
        if "ALERT" in statuses:
            overall_status = "ALERT"
        elif "WARNING" in statuses:
            overall_status = "WARNING"

        return {
            "timestamp": datetime.now().isoformat(),
            "total_predictions": len(df),
            "reference_period": f"First {split_idx} predictions",
            "production_period": f"Last {len(production_df)} predictions",
            "drift_analysis": drift_results,
            "overall_status": overall_status,
            "recommendations": self._get_recommendations(overall_status)
        }

    def _get_recommendations(self, status: str) -> List[str]:
        """
        Generate recommendations based on drift status.

        Args:
            status: Overall drift status

        Returns:
            List of recommended actions
        """
        if status == "ALERT":
            return [
                "Immediate investigation required - significant data drift detected",
                "Consider retraining the model with recent data",
                "Monitor prediction performance closely",
                "Check for changes in customer behavior or data collection"
            ]
        elif status == "WARNING":
            return [
                "Monitor drift trends closely",
                "Consider gradual model updates",
                "Validate prediction quality on recent data"
            ]
        else:
            return [
                "System operating normally",
                "Continue regular monitoring"
            ]


# Convenience functions for external use
def log_prediction(customer_id: str, service_id: str, score: float,
                  urgency: int, timestamp: str = None) -> None:
    """
    Log a prediction to the monitoring system.

    Args:
        customer_id: Unique customer identifier
        service_id: Recommended service identifier
        score: Recommendation score (0-1)
        urgency: Urgency score (1-10)
        timestamp: ISO format timestamp (auto-generated if None)
    """
    detector = DriftDetector()
    detector.log_prediction(customer_id, service_id, score, urgency, timestamp)


def detect_feature_drift(reference_df: pd.DataFrame,
                        production_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect drift in customer features.

    Args:
        reference_df: Reference dataset (historical)
        production_df: Production dataset (recent predictions)

    Returns:
        Dictionary with drift scores and status for each feature
    """
    detector = DriftDetector()
    return detector.detect_feature_drift(reference_df, production_df)


def get_drift_report() -> Dict[str, Any]:
    """
    Generate a comprehensive drift report.

    Returns:
        Dictionary containing drift analysis results
    """
    detector = DriftDetector()
    return detector.get_drift_report()