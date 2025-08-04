"""
Data models for the call analysis system.

This module contains both Pydantic models for API serialization and SQLAlchemy models
for database persistence.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field, ConfigDict

Base = declarative_base()

# Pydantic Models for API
class CallInsightBase(BaseModel):
    """Base model for call insights."""
    model_config = ConfigDict(from_attributes=True)
    
    call_id: str = Field(..., description="Unique identifier for the call")
    clinic_mentioned: str = Field(default="unknown", description="Name of clinic mentioned")
    primary_intent: str = Field(..., description="Primary purpose of the call")
    secondary_intents: List[str] = Field(default_factory=list, description="Secondary purposes")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1 to 1")
    emotional_journey: Dict[str, float] = Field(default_factory=dict, description="Emotions throughout call")
    urgency_level: int = Field(..., ge=1, le=5, description="Urgency level from 1-5")
    resolution_status: str = Field(..., description="Status of call resolution")
    staff_performance: Dict[str, float] = Field(default_factory=dict, description="Staff performance metrics")
    revenue_opportunity: float = Field(default=0.0, ge=0.0, description="Estimated revenue opportunity")
    service_requests: List[str] = Field(default_factory=list, description="Services requested")
    pain_points: List[str] = Field(default_factory=list, description="Customer pain points")
    success_factors: List[str] = Field(default_factory=list, description="Success factors identified")
    next_actions: List[str] = Field(default_factory=list, description="Recommended next actions")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence in analysis")

class CallInsightCreate(CallInsightBase):
    """Model for creating call insights."""
    pass

class CallInsightUpdate(BaseModel):
    """Model for updating call insights."""
    model_config = ConfigDict(from_attributes=True)
    
    clinic_mentioned: Optional[str] = None
    primary_intent: Optional[str] = None
    secondary_intents: Optional[List[str]] = None
    sentiment_score: Optional[float] = Field(None, ge=-1.0, le=1.0)
    emotional_journey: Optional[Dict[str, float]] = None
    urgency_level: Optional[int] = Field(None, ge=1, le=5)
    resolution_status: Optional[str] = None
    staff_performance: Optional[Dict[str, float]] = None
    revenue_opportunity: Optional[float] = Field(None, ge=0.0)
    service_requests: Optional[List[str]] = None
    pain_points: Optional[List[str]] = None
    success_factors: Optional[List[str]] = None
    next_actions: Optional[List[str]] = None
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)

class CallInsight(CallInsightBase):
    """Complete call insight model with metadata."""
    id: int = Field(..., description="Database ID")
    timestamp: datetime = Field(..., description="When the insight was created")
    created_at: datetime = Field(..., description="Database creation timestamp")
    updated_at: datetime = Field(..., description="Database update timestamp")

# SQLAlchemy Models for Database
class CallInsightDB(Base):
    """Database model for call insights."""
    __tablename__ = "call_insights"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String, unique=True, index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    clinic_mentioned = Column(String, default="unknown")
    primary_intent = Column(String, nullable=False)
    secondary_intents = Column(JSON, default=list)
    sentiment_score = Column(Float, nullable=False)
    emotional_journey = Column(JSON, default=dict)
    urgency_level = Column(Integer, nullable=False)
    resolution_status = Column(String, nullable=False)
    staff_performance = Column(JSON, default=dict)
    revenue_opportunity = Column(Float, default=0.0)
    service_requests = Column(JSON, default=list)
    pain_points = Column(JSON, default=list)
    success_factors = Column(JSON, default=list)
    next_actions = Column(JSON, default=list)
    confidence_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class StaffPerformanceDB(Base):
    """Database model for staff performance metrics."""
    __tablename__ = "staff_performance"

    id = Column(Integer, primary_key=True, index=True)
    staff_id = Column(String, index=True, nullable=False)
    call_id = Column(String, index=True, nullable=False)
    empathy_score = Column(Float, default=0.0)
    efficiency_score = Column(Float, default=0.0)
    knowledge_score = Column(Float, default=0.0)
    professionalism_score = Column(Float, default=0.0)
    overall_score = Column(Float, default=0.0)
    coaching_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class PredictionDB(Base):
    """Database model for storing predictions."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_type = Column(String, nullable=False)  # call_volume, sentiment, revenue
    prediction_date = Column(DateTime, nullable=False)
    predicted_value = Column(Float, nullable=False)
    confidence_score = Column(Float, default=0.0)
    actual_value = Column(Float)  # Filled in later for accuracy tracking
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class CoachingSessionDB(Base):
    """Database model for coaching sessions."""
    __tablename__ = "coaching_sessions"

    id = Column(Integer, primary_key=True, index=True)
    call_id = Column(String, unique=True, index=True, nullable=False)
    staff_id = Column(String, index=True, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime)
    alerts_triggered = Column(JSON, default=list)
    coaching_suggestions = Column(JSON, default=list)
    scripts_used = Column(JSON, default=list)
    escalation_flags = Column(JSON, default=list)
    final_performance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic models for API responses
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    model_config = ConfigDict(from_attributes=True)
    
    daily_forecasts: List[Dict[str, Any]] = Field(..., description="Daily predictions")
    weekly_totals: Dict[str, Any] = Field(..., description="Weekly aggregate predictions")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence in predictions")

class CoachingResponse(BaseModel):
    """Response model for real-time coaching."""
    model_config = ConfigDict(from_attributes=True)
    
    timestamp: str = Field(..., description="Timestamp of coaching response")
    immediate_alerts: List[str] = Field(default_factory=list, description="Urgent alerts")
    coaching_suggestions: List[str] = Field(default_factory=list, description="Coaching suggestions")
    performance_updates: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    script_recommendations: List[str] = Field(default_factory=list, description="Suggested scripts")
    escalation_flags: List[str] = Field(default_factory=list, description="Escalation warnings")

class AnalyticsResponse(BaseModel):
    """Response model for analytics dashboard."""
    model_config = ConfigDict(from_attributes=True)
    
    total_calls: int = Field(..., description="Total number of calls analyzed")
    average_sentiment: float = Field(..., description="Average sentiment score")
    top_intents: List[Dict[str, Any]] = Field(..., description="Most common call intents")
    staff_performance: Dict[str, Dict[str, float]] = Field(..., description="Staff performance metrics")
    revenue_opportunities: float = Field(..., description="Total revenue opportunities identified")
    common_pain_points: List[Dict[str, Any]] = Field(..., description="Most common customer pain points")

# Dataclass for backward compatibility with original code
@dataclass
class CallInsightDataclass:
    """
    Dataclass version of CallInsight for backward compatibility.
    This maintains compatibility with the original implementation.
    """
    call_id: str
    timestamp: datetime
    clinic_mentioned: str
    primary_intent: str
    secondary_intents: List[str]
    sentiment_score: float
    emotional_journey: Dict[str, float]
    urgency_level: int
    resolution_status: str
    staff_performance: Dict[str, float]
    revenue_opportunity: float
    service_requests: List[str]
    pain_points: List[str]
    success_factors: List[str]
    next_actions: List[str]
    confidence_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(self.to_dict(), default=json_serializer, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CallInsightDataclass":
        """Create from dictionary."""
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "CallInsightDataclass":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

# Export the dataclass as CallInsight for backward compatibility
CallInsight = CallInsightDataclass