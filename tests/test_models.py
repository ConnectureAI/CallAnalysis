"""
Tests for data models and database operations.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from src.call_analysis.models import (
    CallInsight, CallInsightDB, CallInsightDataclass,
    StaffPerformanceDB, PredictionDB, CoachingSessionDB
)


class TestCallInsightDataclass:
    """Test CallInsight dataclass functionality."""

    def test_create_call_insight(self, sample_call_insight):
        """Test creating a CallInsight instance."""
        insight = sample_call_insight
        
        assert insight.call_id == "test_call_001"
        assert insight.primary_intent == "appointment_booking"
        assert insight.sentiment_score == 0.8
        assert insight.urgency_level == 2
        assert insight.confidence_score == 0.85

    def test_call_insight_to_dict(self, sample_call_insight):
        """Test converting CallInsight to dictionary."""
        insight_dict = sample_call_insight.to_dict()
        
        assert isinstance(insight_dict, dict)
        assert insight_dict["call_id"] == "test_call_001"
        assert insight_dict["primary_intent"] == "appointment_booking"
        assert insight_dict["sentiment_score"] == 0.8

    def test_call_insight_to_json(self, sample_call_insight):
        """Test converting CallInsight to JSON."""
        insight_json = sample_call_insight.to_json()
        
        assert isinstance(insight_json, str)
        assert "test_call_001" in insight_json
        assert "appointment_booking" in insight_json

    def test_call_insight_from_dict(self, sample_call_insight):
        """Test creating CallInsight from dictionary."""
        insight_dict = sample_call_insight.to_dict()
        new_insight = CallInsightDataclass.from_dict(insight_dict)
        
        assert new_insight.call_id == sample_call_insight.call_id
        assert new_insight.primary_intent == sample_call_insight.primary_intent
        assert new_insight.sentiment_score == sample_call_insight.sentiment_score

    def test_call_insight_from_json(self, sample_call_insight):
        """Test creating CallInsight from JSON."""
        insight_json = sample_call_insight.to_json()
        new_insight = CallInsightDataclass.from_json(insight_json)
        
        assert new_insight.call_id == sample_call_insight.call_id
        assert new_insight.primary_intent == sample_call_insight.primary_intent


@pytest.mark.asyncio
class TestDatabaseModels:
    """Test database model operations."""

    async def test_create_call_insight_db(self, db_session):
        """Test creating CallInsightDB instance."""
        db_insight = CallInsightDB(
            call_id="db_test_001",
            timestamp=datetime.now(),
            clinic_mentioned="Test Clinic",
            primary_intent="appointment_booking",
            secondary_intents=["inquiry"],
            sentiment_score=0.8,
            emotional_journey={"beginning": 0.5, "middle": 0.8, "end": 0.9},
            urgency_level=2,
            resolution_status="resolved",
            staff_performance={"empathy": 0.9},
            revenue_opportunity=250.0,
            service_requests=["cleaning"],
            pain_points=[],
            success_factors=["friendly_staff"],
            next_actions=["schedule_appointment"],
            confidence_score=0.85
        )
        
        db_session.add(db_insight)
        await db_session.commit()
        
        # Verify it was saved
        assert db_insight.id is not None
        assert db_insight.call_id == "db_test_001"

    async def test_query_call_insights(self, db_with_sample_data):
        """Test querying call insights from database."""
        from sqlalchemy import select
        
        stmt = select(CallInsightDB).where(CallInsightDB.call_id == "test_call_001")
        result = await db_with_sample_data.execute(stmt)
        insight = result.scalar_one_or_none()
        
        assert insight is not None
        assert insight.call_id == "test_call_001"
        assert insight.primary_intent == "appointment_booking"

    async def test_create_staff_performance_db(self, db_session):
        """Test creating StaffPerformanceDB instance."""
        staff_perf = StaffPerformanceDB(
            staff_id="staff_001",
            call_id="call_001",
            empathy_score=0.9,
            efficiency_score=0.8,
            knowledge_score=0.8,
            professionalism_score=0.9,
            overall_score=0.85,
            coaching_notes="Excellent customer rapport"
        )
        
        db_session.add(staff_perf)
        await db_session.commit()
        
        assert staff_perf.id is not None
        assert staff_perf.staff_id == "staff_001"

    async def test_create_prediction_db(self, db_session):
        """Test creating PredictionDB instance."""
        prediction = PredictionDB(
            prediction_type="call_volume",
            prediction_date=datetime.now(),
            predicted_value=25.0,
            confidence_score=0.8,
            model_version="v1.0"
        )
        
        db_session.add(prediction)
        await db_session.commit()
        
        assert prediction.id is not None
        assert prediction.prediction_type == "call_volume"

    async def test_create_coaching_session_db(self, db_session):
        """Test creating CoachingSessionDB instance."""
        coaching_session = CoachingSessionDB(
            call_id="coaching_test_001",
            staff_id="staff_001",
            start_time=datetime.now(),
            alerts_triggered=["sentiment_drop"],
            coaching_suggestions=["use_empathy"],
            scripts_used=["de_escalation"],
            escalation_flags=[],
            final_performance_score=0.8
        )
        
        db_session.add(coaching_session)
        await db_session.commit()
        
        assert coaching_session.id is not None
        assert coaching_session.call_id == "coaching_test_001"


class TestPydanticModels:
    """Test Pydantic model validation."""

    def test_call_insight_base_validation(self):
        """Test CallInsightBase validation."""
        from src.call_analysis.models import CallInsightBase
        
        # Valid data
        valid_data = {
            "call_id": "test_001",
            "primary_intent": "appointment_booking",
            "sentiment_score": 0.8,
            "urgency_level": 2,
            "resolution_status": "resolved",
            "confidence_score": 0.85
        }
        
        insight = CallInsightBase(**valid_data)
        assert insight.call_id == "test_001"
        assert insight.sentiment_score == 0.8

    def test_call_insight_base_validation_errors(self):
        """Test CallInsightBase validation with invalid data."""
        from src.call_analysis.models import CallInsightBase
        from pydantic import ValidationError
        
        # Invalid sentiment score (out of range)
        with pytest.raises(ValidationError):
            CallInsightBase(
                call_id="test_001",
                primary_intent="appointment_booking",
                sentiment_score=2.0,  # Should be between -1 and 1
                urgency_level=2,
                resolution_status="resolved",
                confidence_score=0.85
            )
        
        # Invalid urgency level (out of range)
        with pytest.raises(ValidationError):
            CallInsightBase(
                call_id="test_001",
                primary_intent="appointment_booking",
                sentiment_score=0.8,
                urgency_level=10,  # Should be between 1 and 5
                resolution_status="resolved",
                confidence_score=0.85
            )

    def test_prediction_response_model(self):
        """Test PredictionResponse model."""
        from src.call_analysis.models import PredictionResponse
        
        prediction_data = {
            "daily_forecasts": [
                {"date": "2025-01-01", "call_volume": 25, "sentiment": 0.8}
            ],
            "weekly_totals": {"total_calls": 175, "avg_sentiment": 0.75},
            "recommendations": ["Schedule extra staff on Monday"]
        }
        
        response = PredictionResponse(**prediction_data)
        assert len(response.daily_forecasts) == 1
        assert response.weekly_totals["total_calls"] == 175

    def test_coaching_response_model(self):
        """Test CoachingResponse model."""
        from src.call_analysis.models import CoachingResponse
        
        coaching_data = {
            "timestamp": "2025-01-01T10:00:00",
            "immediate_alerts": ["sentiment_drop"],
            "coaching_suggestions": ["use_empathy"],
            "script_recommendations": ["de_escalation_script"]
        }
        
        response = CoachingResponse(**coaching_data)
        assert response.timestamp == "2025-01-01T10:00:00"
        assert len(response.immediate_alerts) == 1


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for models with database."""

    @pytest.mark.asyncio
    async def test_full_model_workflow(self, db_session, sample_call_insight):
        """Test complete model workflow from creation to query."""
        from sqlalchemy import select
        
        # 1. Create database model from dataclass
        db_insight = CallInsightDB(
            call_id=sample_call_insight.call_id,
            timestamp=sample_call_insight.timestamp,
            clinic_mentioned=sample_call_insight.clinic_mentioned,
            primary_intent=sample_call_insight.primary_intent,
            secondary_intents=sample_call_insight.secondary_intents,
            sentiment_score=sample_call_insight.sentiment_score,
            emotional_journey=sample_call_insight.emotional_journey,
            urgency_level=sample_call_insight.urgency_level,
            resolution_status=sample_call_insight.resolution_status,
            staff_performance=sample_call_insight.staff_performance,
            revenue_opportunity=sample_call_insight.revenue_opportunity,
            service_requests=sample_call_insight.service_requests,
            pain_points=sample_call_insight.pain_points,
            success_factors=sample_call_insight.success_factors,
            next_actions=sample_call_insight.next_actions,
            confidence_score=sample_call_insight.confidence_score
        )
        
        # 2. Save to database
        db_session.add(db_insight)
        await db_session.commit()
        
        # 3. Query back from database
        stmt = select(CallInsightDB).where(CallInsightDB.call_id == sample_call_insight.call_id)
        result = await db_session.execute(stmt)
        retrieved_insight = result.scalar_one()
        
        # 4. Verify data integrity
        assert retrieved_insight.call_id == sample_call_insight.call_id
        assert retrieved_insight.primary_intent == sample_call_insight.primary_intent
        assert retrieved_insight.sentiment_score == sample_call_insight.sentiment_score
        assert retrieved_insight.secondary_intents == sample_call_insight.secondary_intents
        assert retrieved_insight.staff_performance == sample_call_insight.staff_performance