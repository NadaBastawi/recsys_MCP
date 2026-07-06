"""
TORCO Recommendation System - Data Quality Tests

Tests for ensuring data integrity and compliance with business rules.
"""

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    data_dir = 'data'
    customers = pd.read_csv(f'{data_dir}/customers.csv')
    interactions = pd.read_csv(f'{data_dir}/interactions.csv')
    services = pd.read_csv(f'{data_dir}/services.csv')
    return customers, interactions, services


class TestDataQuality:
    """Test suite for data quality validation."""

    def test_customers_no_null_required_fields(self, sample_data):
        """Test that required customer fields have no null values."""
        customers, _, _ = sample_data
        required_fields = ['customer_id', 'zip_code', 'property_type', 'building_age_years']
        
        for field in required_fields:
            assert field in customers.columns, f"Missing field: {field}"
            assert customers[field].isnull().sum() == 0, f"Null values found in {field}"

    def test_interactions_no_null_required_fields(self, sample_data):
        """Test that required interaction fields have no null values."""
        _, interactions, _ = sample_data
        required_fields = ['interaction_id', 'customer_id', 'service_id', 'rating', 'urgency_score']
        
        for field in required_fields:
            assert field in interactions.columns, f"Missing field: {field}"
            assert interactions[field].isnull().sum() == 0, f"Null values found in {field}"

    def test_valid_zip_codes(self, sample_data):
        """Test that ZIP codes are valid (numeric, 5 digits)."""
        customers, _, _ = sample_data
        
        # ZIP codes can be int64 or string
        assert customers['zip_code'].dtype in [object, 'int64'], \
            f"ZIP codes have unexpected dtype: {customers['zip_code'].dtype}"
        
        for zip_code in customers['zip_code'].unique():
            zip_str = str(zip_code)
            assert len(zip_str) == 5, f"Invalid ZIP code length: {zip_code}"
            assert zip_str.isdigit(), f"Invalid ZIP code format: {zip_code}"

    def test_rating_range(self, sample_data):
        """Test that ratings are in valid range [1, 5]."""
        _, interactions, _ = sample_data
        
        assert interactions['rating'].min() >= 1, "Rating below minimum (1)"
        assert interactions['rating'].max() <= 5, "Rating above maximum (5)"
        assert (interactions['rating'] % 1 == 0).all(), "Ratings should be integers"

    def test_urgency_score_range(self, sample_data):
        """Test that urgency scores are in valid range [1, 10]."""
        _, interactions, _ = sample_data
        
        assert interactions['urgency_score'].min() >= 1, "Urgency score below minimum (1)"
        assert interactions['urgency_score'].max() <= 10, "Urgency score above maximum (10)"
        assert (interactions['urgency_score'] % 1 == 0).all(), "Urgency scores should be integers"

    def test_valid_service_ids(self, sample_data):
        """Test that all service IDs are valid and exist in service catalog."""
        _, interactions, services = sample_data
        
        valid_services = set(services['service_id'].unique())
        interaction_services = set(interactions['service_id'].unique())
        
        assert interaction_services.issubset(valid_services), \
            f"Found invalid service IDs: {interaction_services - valid_services}"

    def test_valid_customer_ids(self, sample_data):
        """Test that all customer IDs in interactions exist in customer table."""
        customers, interactions, _ = sample_data
        
        valid_customers = set(customers['customer_id'].unique())
        interaction_customers = set(interactions['customer_id'].unique())
        
        assert interaction_customers.issubset(valid_customers), \
            f"Found invalid customer IDs: {interaction_customers - valid_customers}"

    def test_building_age_reasonable(self, sample_data):
        """Test that building ages are in reasonable range."""
        customers, _, _ = sample_data
        
        assert customers['building_age_years'].min() >= 0, "Building age cannot be negative"
        assert customers['building_age_years'].max() <= 200, "Building age seems unreasonable"

    def test_customer_uniqueness(self, sample_data):
        """Test that customer IDs are unique."""
        customers, _, _ = sample_data
        
        assert len(customers['customer_id']) == len(customers['customer_id'].unique()), \
            "Duplicate customer IDs found"

    def test_interaction_uniqueness(self, sample_data):
        """Test that interaction IDs are unique."""
        _, interactions, _ = sample_data
        
        assert len(interactions['interaction_id']) == len(interactions['interaction_id'].unique()), \
            "Duplicate interaction IDs found"

    def test_valid_property_types(self, sample_data):
        """Test that property types are from predefined set."""
        customers, _, _ = sample_data
        
        valid_types = {
            'residential_house', 'residential_apartment',
            'commercial_restaurant', 'commercial_office', 'commercial_warehouse',
            'mobile_home'  # Additional property type in synthetic data
        }
        actual_types = set(customers['property_type'].unique())
        
        assert actual_types.issubset(valid_types), \
            f"Found invalid property types: {actual_types - valid_types}"

    def test_valid_seasons(self, sample_data):
        """Test that seasons are valid."""
        _, interactions, _ = sample_data
        
        valid_seasons = {'winter', 'spring', 'summer', 'fall'}
        actual_seasons = set(interactions['season'].unique())
        
        assert actual_seasons.issubset(valid_seasons), \
            f"Found invalid seasons: {actual_seasons - valid_seasons}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
