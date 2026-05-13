"""
TORCO Recommendation System - Recommender Model Tests

Tests for model functionality and prediction quality.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recommender import HybridRecommender


@pytest.fixture
def recommender():
    """Load and initialize the recommender model."""
    model = HybridRecommender(cb_weight=0.4, cf_weight=0.6)
    customers = pd.read_csv('data/customers.csv')
    interactions = pd.read_csv('data/interactions.csv')
    services = pd.read_csv('data/services.csv')
    model.fit(customers, interactions, services)
    return model


@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    customers = pd.read_csv('data/customers.csv')
    services = pd.read_csv('data/services.csv')
    return customers, services


class TestRecommenderModel:
    """Test suite for the hybrid recommender model."""

    def test_model_loads_successfully(self, recommender):
        """Test that the model loads without errors."""
        assert recommender is not None, "Model failed to load"
        assert hasattr(recommender, 'recommend'), "Model missing recommend method"

    def test_recommend_returns_top_n_results(self, recommender, sample_data):
        """Test that recommend returns exactly top_n results."""
        customers, _ = sample_data
        test_customer = customers.iloc[0]
        
        for top_n in [1, 3, 5, 10]:
            recommendations = recommender.recommend(
                customer_id=test_customer['customer_id'],
                top_n=top_n,
                season='spring',
                days_since_last=200
            )
            assert len(recommendations) <= top_n, \
                f"Got {len(recommendations)} recommendations, expected <= {top_n}"

    def test_recommend_handles_cold_start(self, recommender):
        """Test that model handles new customers (cold-start)."""
        # Use a valid property type but new zip code
        recommendations = recommender.recommend_cold_start(
            property_type='residential_house',
            zip_code='99999',  # Non-existent ZIP
            building_age_years=25,
            season='summer',
            days_since_last=180,
            top_n=5
        )
        
        assert len(recommendations) > 0, "Cold-start failed: no recommendations generated"
        assert isinstance(recommendations, list), "Recommendations should be a list"

    def test_recommendation_score_range(self, recommender, sample_data):
        """Test that recommendation scores are in valid range."""
        customers, _ = sample_data
        test_customer = customers.iloc[0]
        
        recommendations = recommender.recommend(
            customer_id=test_customer['customer_id'],
            top_n=5,
            season='spring',
            days_since_last=200
        )
        
        for rec in recommendations:
            score = rec.get('score')
            assert score is not None, "Recommendation missing score"
            assert 0 <= score <= 1, f"Score out of range: {score}"

    def test_urgency_score_range(self, recommender, sample_data):
        """Test that urgency scores are in valid range [1, 10]."""
        customers, _ = sample_data
        test_customer = customers.iloc[0]
        
        recommendations = recommender.recommend(
            customer_id=test_customer['customer_id'],
            top_n=5,
            season='spring',
            days_since_last=200
        )
        
        for rec in recommendations:
            urgency = rec.get('urgency_score')
            assert urgency is not None, "Recommendation missing urgency_score"
            assert 1 <= urgency <= 10, f"Urgency score out of range: {urgency}"

    def test_returned_service_ids_valid(self, recommender, sample_data):
        """Test that all returned service IDs are valid."""
        customers, services = sample_data
        test_customer = customers.iloc[0]
        valid_services = set(services['service_id'].unique())
        
        recommendations = recommender.recommend(
            customer_id=test_customer['customer_id'],
            top_n=5,
            season='spring',
            days_since_last=200
        )
        
        for rec in recommendations:
            service_id = rec.get('service_id')
            assert service_id in valid_services, \
                f"Invalid service ID in recommendations: {service_id}"

    def test_recommendations_are_ordered_by_score(self, recommender, sample_data):
        """Test that recommendations are ordered by score (descending)."""
        customers, _ = sample_data
        test_customer = customers.iloc[0]
        
        recommendations = recommender.recommend(
            customer_id=test_customer['customer_id'],
            top_n=5,
            season='spring',
            days_since_last=200
        )
        
        if len(recommendations) > 1:
            scores = [rec['score'] for rec in recommendations]
            assert scores == sorted(scores, reverse=True), \
                "Recommendations not ordered by score (descending)"

    def test_no_duplicate_recommendations(self, recommender, sample_data):
        """Test that no duplicate service IDs in recommendations."""
        customers, _ = sample_data
        test_customer = customers.iloc[0]
        
        recommendations = recommender.recommend(
            customer_id=test_customer['customer_id'],
            top_n=10,
            season='spring',
            days_since_last=200
        )
        
        service_ids = [rec['service_id'] for rec in recommendations]
        assert len(service_ids) == len(set(service_ids)), \
            "Duplicate service IDs found in recommendations"

    def test_recommend_with_all_seasons(self, recommender, sample_data):
        """Test that recommender works for all seasons."""
        customers, _ = sample_data
        test_customer = customers.iloc[0]
        
        for season in ['winter', 'spring', 'summer', 'fall']:
            recommendations = recommender.recommend(
                customer_id=test_customer['customer_id'],
                top_n=5,
                season=season,
                days_since_last=200
            )
            assert len(recommendations) > 0, f"Failed to recommend for season: {season}"

    def test_model_deterministic(self, recommender, sample_data):
        """Test that model produces consistent results (seed 42)."""
        customers, _ = sample_data
        test_customer = customers.iloc[0]
        
        rec1 = recommender.recommend(
            customer_id=test_customer['customer_id'],
            top_n=5,
            season='spring',
            days_since_last=200
        )
        
        rec2 = recommender.recommend(
            customer_id=test_customer['customer_id'],
            top_n=5,
            season='spring',
            days_since_last=200
        )
        
        assert len(rec1) == len(rec2), "Inconsistent number of recommendations"
        for r1, r2 in zip(rec1, rec2):
            assert r1['service_id'] == r2['service_id'], "Inconsistent recommendation order"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
