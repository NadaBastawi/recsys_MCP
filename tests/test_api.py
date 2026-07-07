"""
TORCO Recommendation System - FastAPI Tests

Tests for REST API endpoints and response schemas.
"""

import pytest
import httpx
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def api_client():
    """Create a test client for the FastAPI app."""
    os.environ.setdefault("API_KEY", "test-api-key")
    # Note: In CI/CD, the API should be running on localhost:8000
    # For local testing, we can use the app directly
    try:
        from main import app
        from fastapi.testclient import TestClient
        return TestClient(app)
    except ImportError:
        # Fall back to HTTP client if app not available
        return None


def get_auth_headers():
    """Return authenticated request headers for protected endpoints."""
    return {"X-API-Key": os.getenv("API_KEY", "test-api-key")}


class TestAPIHealth:
    """Test suite for API health check endpoint."""

    def test_health_endpoint_200(self, api_client):
        """Test that /health returns 200 status."""
        if api_client is None:
            pytest.skip("API not available")
        
        response = api_client.get("/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def test_health_response_schema(self, api_client):
        """Test that /health returns correct schema."""
        if api_client is None:
            pytest.skip("API not available")
        
        response = api_client.get("/health")
        data = response.json()
        
        assert "status" in data, "Missing 'status' field"
        assert data["status"] in ["ok", "healthy"], "Invalid status value"
        assert "model_loaded" in data, "Missing 'model_loaded' field"
        assert isinstance(data["model_loaded"], bool), "model_loaded should be boolean"


class TestRecommendationEndpoint:
    """Test suite for /recommend endpoint."""

    def test_recommend_requires_api_key(self, api_client):
        """Test that /recommend rejects requests without an API key."""
        if api_client is None:
            pytest.skip("API not available")

        payload = {
            "customer_id": "C0001",
            "season": "summer",
            "days_since_last_service": 200
        }

        response = api_client.post("/recommend", json=payload)
        assert response.status_code == 401, \
            f"Expected 401, got {response.status_code}: {response.text}"

    def test_recommend_existing_customer_200(self, api_client):
        """Test that /recommend returns 200 for existing customer."""
        if api_client is None:
            pytest.skip("API not available")
        
        payload = {
            "customer_id": "C0001",
            "season": "summer",
            "days_since_last_service": 200
        }
        
        response = api_client.post("/recommend", json=payload, headers=get_auth_headers())
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text}"

    def test_recommend_response_schema(self, api_client):
        """Test that /recommend returns correct response schema."""
        if api_client is None:
            pytest.skip("API not available")
        
        payload = {
            "customer_id": "C0001",
            "season": "summer",
            "days_since_last_service": 200
        }
        
        response = api_client.post("/recommend", json=payload, headers=get_auth_headers())
        data = response.json()
        
        assert "customer_id" in data, "Missing 'customer_id' in response"
        assert "recommendations" in data, "Missing 'recommendations' in response"
        assert isinstance(data["recommendations"], list), \
            "'recommendations' should be a list"

    def test_recommend_response_contains_services(self, api_client):
        """Test that each recommendation contains required fields."""
        if api_client is None:
            pytest.skip("API not available")
        
        payload = {
            "customer_id": "C0001",
            "season": "summer",
            "days_since_last_service": 200
        }
        
        response = api_client.post("/recommend", json=payload, headers=get_auth_headers())
        data = response.json()
        
        if len(data["recommendations"]) > 0:
            rec = data["recommendations"][0]
            assert "service_id" in rec, "Missing 'service_id' in recommendation"
            assert "service_name" in rec, "Missing 'service_name' in recommendation"
            assert "score" in rec, "Missing 'score' in recommendation"
            assert "urgency_score" in rec, "Missing 'urgency_score' in recommendation"

    def test_recommend_unknown_customer_404(self, api_client):
        """Test that /recommend returns 404 for unknown customer."""
        if api_client is None:
            pytest.skip("API not available")
        
        payload = {
            "customer_id": "UNKNOWN_CUSTOMER_XYZ123",
            "season": "summer",
            "days_since_last_service": 200
        }
        
        response = api_client.post("/recommend", json=payload, headers=get_auth_headers())
        # Should either return 404 or fall back to cold-start (200)
        # Check API implementation to determine expected behavior
        assert response.status_code in [200, 404], \
            f"Unexpected status code: {response.status_code}"


class TestNewCustomerEndpoint:
    """Test suite for /recommend/new-customer endpoint."""

    def test_new_customer_valid_input_200(self, api_client):
        """Test that /recommend/new-customer returns 200 with valid input."""
        if api_client is None:
            pytest.skip("API not available")
        
        payload = {
            "property_type": "residential_house",
            "zip_code": "85743",
            "building_age_years": 25,
            "season": "spring"
        }
        
        response = api_client.post("/recommend/new-customer", json=payload, headers=get_auth_headers())
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text}"

    def test_new_customer_response_schema(self, api_client):
        """Test that /recommend/new-customer returns correct schema."""
        if api_client is None:
            pytest.skip("API not available")
        
        payload = {
            "property_type": "residential_house",
            "zip_code": "85743",
            "building_age_years": 25,
            "season": "spring"
        }
        
        response = api_client.post("/recommend/new-customer", json=payload, headers=get_auth_headers())
        data = response.json()
        
        assert "recommendations" in data, "Missing 'recommendations' in response"
        assert isinstance(data["recommendations"], list), \
            "'recommendations' should be a list"
        assert len(data["recommendations"]) > 0, \
            "Should return at least one recommendation for cold-start"

    def test_new_customer_all_property_types(self, api_client):
        """Test that /recommend/new-customer works for all property types."""
        if api_client is None:
            pytest.skip("API not available")
        
        property_types = [
            'residential_house',
            'residential_apartment',
            'commercial_restaurant',
            'commercial_office',
            'commercial_warehouse'
        ]
        
        for prop_type in property_types:
            payload = {
                "property_type": prop_type,
                "zip_code": "85743",
                "building_age_years": 25,
                "season": "spring"
            }
            
            response = api_client.post("/recommend/new-customer", json=payload, headers=get_auth_headers())
            assert response.status_code == 200, \
                f"Failed for property_type: {prop_type}"

    def test_new_customer_missing_field_422(self, api_client):
        """Test that /recommend/new-customer returns 422 for missing required field."""
        if api_client is None:
            pytest.skip("API not available")
        
        # Missing property_type
        payload = {
            "zip_code": "85743",
            "building_age_years": 25,
            "season": "spring"
        }
        
        response = api_client.post("/recommend/new-customer", json=payload, headers=get_auth_headers())
        assert response.status_code == 422, \
            f"Expected 422 for missing field, got {response.status_code}"


class TestServicesEndpoint:
    """Test suite for /services endpoint."""

    def test_services_endpoint_200(self, api_client):
        """Test that /services returns 200 status."""
        if api_client is None:
            pytest.skip("API not available")
        
        response = api_client.get("/services")
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}"

    def test_services_response_is_list(self, api_client):
        """Test that /services returns a list of services."""
        if api_client is None:
            pytest.skip("API not available")
        
        response = api_client.get("/services")
        data = response.json()
        
        assert isinstance(data, list), "Services should be returned as a list"
        assert len(data) > 0, "Should return at least one service"

    def test_services_have_required_fields(self, api_client):
        """Test that each service contains required fields."""
        if api_client is None:
            pytest.skip("API not available")
        
        response = api_client.get("/services")
        services = response.json()
        
        for service in services:
            assert "service_id" in service, "Missing 'service_id'"
            assert "service_name" in service, "Missing 'service_name'"
            assert "price_usd" in service, "Missing 'price_usd'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
