"""
Pytest configuration and shared fixtures for Call Analysis System tests.
"""

import asyncio
import os
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from typing import AsyncGenerator

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import event

from src.call_analysis.api import app
from src.call_analysis.config import Settings, get_settings
from src.call_analysis.database import get_db_session_dependency
from src.call_analysis.models import Base, CallInsight
from src.call_analysis.analyzer import SemanticTranscriptAnalyzer
from src.call_analysis.predictor import PredictiveAnalytics
from src.call_analysis.coaching import RealTimeCoachingSystem


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Create test settings."""
    return Settings(
        environment="testing",
        debug=True,
        database={"url": TEST_DATABASE_URL},
        openai={"api_key": "sk-test-fake-key", "model": "gpt-3.5-turbo"},
        security={"secret_key": "test-secret-key-for-testing-only"},
    )


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
def override_get_db(db_session):
    """Override database dependency for testing."""
    async def _override_get_db():
        yield db_session
    
    app.dependency_overrides[get_db_session_dependency] = _override_get_db
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def override_get_settings(test_settings):
    """Override settings dependency for testing."""
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def client(override_get_db, override_get_settings):
    """Create test client."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = AsyncMock()
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = """{
        "primary_intent": "appointment_booking",
        "secondary_intents": ["inquiry"],
        "sentiment_score": 0.8,
        "emotional_journey": {"beginning": 0.5, "middle": 0.8, "end": 0.9},
        "urgency_level": 2,
        "resolution_status": "resolved",
        "staff_performance": {"empathy": 0.9, "efficiency": 0.8, "knowledge": 0.8, "professionalism": 0.9},
        "revenue_opportunity": 250.0,
        "service_requests": ["cleaning"],
        "pain_points": [],
        "success_factors": ["friendly_staff"],
        "next_actions": ["schedule_appointment"],
        "confidence_score": 0.85,
        "clinic_mentioned": "test_clinic"
    }"""
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def analyzer(mock_openai_client):
    """Create test analyzer with mocked OpenAI client."""
    analyzer = SemanticTranscriptAnalyzer(api_key="test-key")
    analyzer.client = mock_openai_client
    return analyzer


@pytest.fixture
def sample_call_insight():
    """Create a sample CallInsight for testing."""
    return CallInsight(
        call_id="test_call_001",
        timestamp=datetime.now(),
        clinic_mentioned="Test Clinic",
        primary_intent="appointment_booking",
        secondary_intents=["inquiry"],
        sentiment_score=0.8,
        emotional_journey={"beginning": 0.5, "middle": 0.8, "end": 0.9},
        urgency_level=2,
        resolution_status="resolved",
        staff_performance={"empathy": 0.9, "efficiency": 0.8, "knowledge": 0.8, "professionalism": 0.9},
        revenue_opportunity=250.0,
        service_requests=["cleaning"],
        pain_points=[],
        success_factors=["friendly_staff"],
        next_actions=["schedule_appointment"],
        confidence_score=0.85
    )


@pytest.fixture
def sample_transcript():
    """Sample call transcript for testing."""
    return """
    Staff: Good morning, thank you for calling Sunny Dental. This is Sarah, how can I help you today?
    
    Customer: Hi Sarah, I'd like to schedule a cleaning appointment. It's been about six months since my last visit.
    
    Staff: I'd be happy to help you schedule that cleaning. Can I get your name and date of birth please?
    
    Customer: Sure, it's John Smith, and my birthday is March 15th, 1985.
    
    Staff: Perfect, I have you in our system. Let me check our availability. We have openings next Tuesday at 2 PM or Thursday at 10 AM. Which works better for you?
    
    Customer: Thursday at 10 AM would be great.
    
    Staff: Excellent! I've scheduled you for Thursday, November 16th at 10 AM with Dr. Johnson for your cleaning. You should receive a confirmation text shortly. Is there anything else I can help you with today?
    
    Customer: No, that's perfect. Thank you so much, Sarah!
    
    Staff: You're very welcome! We look forward to seeing you Thursday. Have a great day!
    """


@pytest.fixture
def predictor():
    """Create test predictor."""
    return PredictiveAnalytics()


@pytest.fixture
def coaching_system():
    """Create test coaching system."""
    return RealTimeCoachingSystem()


# Pytest markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


# Async test configuration
@pytest_asyncio.fixture
async def async_client():
    """Async test client for testing async endpoints."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for all tests."""
    test_env_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "OPENAI_API_KEY": "sk-test-fake-key-for-testing",
        "SECURITY_SECRET_KEY": "test-secret-key-for-testing-only",
        "DB_HOST": "localhost",
        "DB_NAME": "test_db",
        "DB_USER": "test_user",
        "DB_PASSWORD": "test_password",
    }
    
    for key, value in test_env_vars.items():
        monkeypatch.setenv(key, value)


# Database fixtures for different test scenarios
@pytest.fixture
async def db_with_sample_data(db_session, sample_call_insight):
    """Database session with sample data for testing."""
    from src.call_analysis.models import CallInsightDB
    
    # Convert CallInsight to database model
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
    
    db_session.add(db_insight)
    await db_session.commit()
    
    yield db_session


# Performance testing fixtures
@pytest.fixture
def benchmark_data():
    """Generate data for performance benchmarks."""
    return {
        "transcripts": [
            f"Test transcript {i} with some sample content for analysis"
            for i in range(100)
        ],
        "call_ids": [f"call_{i:03d}" for i in range(100)]
    }