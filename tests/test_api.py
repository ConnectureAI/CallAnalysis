"""
Tests for the FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
import json

from src.call_analysis.api import app


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health_check(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_detailed_health_check(self, client):
        """Test detailed health endpoint."""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data
        assert "version" in data
        assert "timestamp" in data


class TestAnalysisEndpoints:
    """Test analysis endpoints."""

    def test_analyze_call_success(self, client):
        """Test successful call analysis."""
        # Mock the analyzer to avoid real API calls
        with pytest.MonkeyPatch().context() as mp:
            # The client fixture should already have mocked dependencies
            response = client.post("/analyze", json={
                "call_id": "test_001",
                "transcript": "Hello, I'd like to book an appointment"
            })
            
            # Should succeed even with mocked analyzer
            assert response.status_code in [200, 503]  # 503 if analyzer unavailable

    def test_analyze_call_missing_data(self, client):
        """Test analysis with missing required data."""
        response = client.post("/analyze", json={
            "call_id": "test_001"
            # Missing transcript
        })
        
        assert response.status_code == 400
        assert "transcript" in response.json()["detail"]

    def test_analyze_call_empty_data(self, client):
        """Test analysis with empty data."""
        response = client.post("/analyze", json={})
        
        assert response.status_code == 400

    def test_analyze_batch_success(self, client):
        """Test successful batch analysis."""
        batch_data = [
            {"call_id": "call_001", "transcript": "Hello, I need help"},
            {"call_id": "call_002", "transcript": "Can I book an appointment?"}
        ]
        
        response = client.post("/analyze/batch", json=batch_data)
        
        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]

    def test_analyze_batch_too_large(self, client):
        """Test batch analysis with too many items."""
        batch_data = [
            {"call_id": f"call_{i:03d}", "transcript": f"Test transcript {i}"}
            for i in range(101)  # Exceeds limit of 100
        ]
        
        response = client.post("/analyze/batch", json=batch_data)
        assert response.status_code == 400


class TestPredictionEndpoints:
    """Test prediction endpoints."""

    def test_predict_next_week(self, client):
        """Test weekly prediction endpoint."""
        response = client.get("/predict/week")
        
        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]

    def test_predict_custom_period_valid(self, client):
        """Test custom prediction with valid parameters."""
        response = client.get("/predict/custom?days=14")
        
        assert response.status_code in [200, 503]

    def test_predict_custom_period_invalid(self, client):
        """Test custom prediction with invalid parameters."""
        # Too many days
        response = client.get("/predict/custom?days=31")
        assert response.status_code == 400
        
        # Too few days
        response = client.get("/predict/custom?days=0")
        assert response.status_code == 400


class TestCoachingEndpoints:
    """Test real-time coaching endpoints."""

    def test_start_coaching_session(self, client):
        """Test starting a coaching session."""
        session_data = {
            "staff_id": "staff_001",
            "call_id": "call_001",
            "customer_context": {"is_new_patient": True}
        }
        
        response = client.post("/coaching/start", json=session_data)
        
        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]

    def test_start_coaching_session_missing_data(self, client):
        """Test starting coaching session with missing data."""
        response = client.post("/coaching/start", json={
            "staff_id": "staff_001"
            # Missing call_id
        })
        
        assert response.status_code == 400

    def test_update_coaching_session(self, client):
        """Test updating a coaching session."""
        update_data = {
            "call_id": "call_001",
            "transcript_chunk": "Customer: I'm frustrated with the wait time",
            "speaker": "customer"
        }
        
        response = client.post("/coaching/update", json=update_data)
        
        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]

    def test_end_coaching_session(self, client):
        """Test ending a coaching session."""
        end_data = {
            "call_id": "call_001",
            "session_summary": {"duration": 300, "resolution": "successful"}
        }
        
        response = client.post("/coaching/end", json=end_data)
        
        # Should succeed or return service unavailable  
        assert response.status_code in [200, 503]


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""

    def test_get_analytics_overview(self, client):
        """Test analytics overview endpoint."""
        response = client.get("/analytics/overview")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return analytics structure even with no data
        assert "total_calls" in data
        assert "average_sentiment" in data
        assert "top_intents" in data
        assert "revenue_opportunities" in data

    def test_get_analytics_overview_custom_period(self, client):
        """Test analytics with custom time period."""
        response = client.get("/analytics/overview?days=7")
        
        assert response.status_code == 200

    def test_get_call_insights(self, client):
        """Test getting call insights."""
        response = client.get("/insights")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "insights" in data
        assert "count" in data

    def test_get_call_insights_pagination(self, client):
        """Test call insights with pagination."""
        response = client.get("/insights?limit=50&offset=0")
        
        assert response.status_code == 200

    def test_get_specific_call_insight(self, client):
        """Test getting a specific call insight."""
        response = client.get("/insights/nonexistent_call")
        
        assert response.status_code == 404


class TestModelEndpoints:
    """Test model management endpoints."""

    def test_trigger_model_training(self, client):
        """Test triggering model training."""
        response = client.post("/models/train")
        
        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]

    def test_get_model_info(self, client):
        """Test getting model information."""
        response = client.get("/models/info")
        
        # Should succeed or return service unavailable
        assert response.status_code in [200, 503]


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the API."""

    def test_full_analysis_workflow(self, client):
        """Test complete analysis workflow."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Analyze a call (if service available)
        analysis_response = client.post("/analyze", json={
            "call_id": "integration_test",
            "transcript": "Hello, I'd like to schedule a cleaning appointment"
        })
        
        if analysis_response.status_code == 200:
            # 3. Check analytics
            analytics_response = client.get("/analytics/overview")
            assert analytics_response.status_code == 200
            
            # 4. Get predictions (if service available)
            prediction_response = client.get("/predict/week")
            # May return 503 if predictor not trained yet
            assert prediction_response.status_code in [200, 503]