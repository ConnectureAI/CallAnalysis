"""
Integration tests for complete workflows.
"""

import pytest
from datetime import datetime, timedelta


@pytest.mark.integration
@pytest.mark.asyncio
class TestFullAnalysisWorkflow:
    """Test complete analysis workflow from transcript to insights."""

    async def test_analyze_to_database_workflow(self, analyzer, db_session, sample_transcript):
        """Test analyzing transcript and saving to database."""
        from src.call_analysis.models import CallInsightDB
        from sqlalchemy import select
        
        # 1. Analyze transcript
        insight = await analyzer.analyze_transcript(sample_transcript, "integration_test_001")
        
        # 2. Convert to database model and save
        db_insight = CallInsightDB(
            call_id=insight.call_id,
            timestamp=insight.timestamp,
            clinic_mentioned=insight.clinic_mentioned,
            primary_intent=insight.primary_intent,
            secondary_intents=insight.secondary_intents,
            sentiment_score=insight.sentiment_score,
            emotional_journey=insight.emotional_journey,
            urgency_level=insight.urgency_level,
            resolution_status=insight.resolution_status,
            staff_performance=insight.staff_performance,
            revenue_opportunity=insight.revenue_opportunity,
            service_requests=insight.service_requests,
            pain_points=insight.pain_points,
            success_factors=insight.success_factors,
            next_actions=insight.next_actions,
            confidence_score=insight.confidence_score
        )
        
        db_session.add(db_insight)
        await db_session.commit()
        
        # 3. Verify data was saved correctly
        stmt = select(CallInsightDB).where(CallInsightDB.call_id == "integration_test_001")
        result = await db_session.execute(stmt)
        saved_insight = result.scalar_one()
        
        assert saved_insight.call_id == insight.call_id
        assert saved_insight.primary_intent == insight.primary_intent
        assert saved_insight.sentiment_score == insight.sentiment_score

    async def test_coaching_session_workflow(self, coaching_system):
        """Test complete real-time coaching workflow."""
        staff_id = "integration_staff_001"
        call_id = "integration_call_001"
        
        # 1. Start coaching session
        start_response = await coaching_system.start_coaching_session(staff_id, call_id)
        assert start_response["status"] == "active"
        assert start_response["session_id"] == call_id
        
        # 2. Process some transcript chunks
        chunks = [
            "Hello, thank you for calling our office.",
            "Customer: Hi, I'm really frustrated with my recent experience.",
            "I understand your frustration. Let me help you with that.",
            "Customer: Thank you, that's much better."
        ]
        
        speakers = ["staff", "customer", "staff", "customer"]
        
        for chunk, speaker in zip(chunks, speakers):
            response = await coaching_system.process_live_transcript_chunk(
                call_id, chunk, speaker
            )
            assert "coaching_suggestions" in response
            assert "immediate_alerts" in response
        
        # 3. End coaching session
        end_response = await coaching_system.end_coaching_session(call_id)
        assert end_response["call_id"] == call_id
        assert "final_performance_metrics" in end_response


@pytest.mark.integration
class TestPredictorWorkflow:
    """Test predictor training and prediction workflow."""

    def test_predictor_training_workflow(self, predictor, sample_call_insight):
        """Test training predictor with sample data."""
        # 1. Add sample insights
        insights = [sample_call_insight] * 60  # Need enough data for training
        
        # Modify timestamps to create realistic data
        for i, insight in enumerate(insights):
            insight.timestamp = datetime.now() - timedelta(days=i)
            insight.call_id = f"training_call_{i:03d}"
        
        predictor.add_insights(insights)
        
        # 2. Train models
        if len(predictor.historical_insights) >= predictor.min_training_samples:
            results = predictor.train_models()
            assert isinstance(results, dict)
            
            # 3. Generate predictions
            if predictor.is_trained:
                predictions = predictor.predict_next_period(7)
                assert "daily_forecasts" in predictions
                assert "weekly_totals" in predictions
                assert "recommendations" in predictions