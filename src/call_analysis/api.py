"""
FastAPI application for the call analysis system.

This module provides the REST API endpoints for interacting with the call analysis
system, including transcript analysis, predictions, real-time coaching, and analytics.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
import structlog

from .config import get_settings
from .database import get_db_session_dependency, init_db, cleanup_db, check_database_health
from .models import (
    CallInsight, CallInsightCreate, CallInsightUpdate,
    PredictionResponse, CoachingResponse, AnalyticsResponse,
    CallInsightDB, StaffPerformanceDB, PredictionDB, CoachingSessionDB
)
from .analyzer import SemanticTranscriptAnalyzer, create_analyzer
from .predictor import PredictiveAnalytics, create_predictor
from .coaching import RealTimeCoachingSystem, create_coaching_system

# Configure structured logging
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    settings = get_settings()
    
    logger.info("Starting Call Analysis API", version=settings.app_version)
    
    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    # Initialize AI components
    try:
        app.state.analyzer = create_analyzer()
        app.state.predictor = create_predictor()
        app.state.coaching_system = create_coaching_system()
        logger.info("AI components initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize AI components", error=str(e))
        # Continue without AI features if they fail
        app.state.analyzer = None
        app.state.predictor = None
        app.state.coaching_system = None
    
    yield
    
    # Cleanup
    logger.info("Shutting down Call Analysis API")
    await cleanup_db()


# Create FastAPI application
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Call Analysis API",
        description="AI-powered dental call center analytics and coaching system",
        version=settings.app_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=settings.security.cors_methods,
        allow_headers=settings.security.cors_headers,
    )
    
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["localhost", "127.0.0.1", settings.host]
        )
    
    return app


app = create_app()


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check including database and AI components."""
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": get_settings().app_version,
        "components": {}
    }
    
    # Check database
    try:
        db_health = await check_database_health()
        health_info["components"]["database"] = db_health
    except Exception as e:
        health_info["components"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_info["status"] = "degraded"
    
    # Check AI components
    try:
        if hasattr(app.state, 'analyzer') and app.state.analyzer:
            api_valid = await app.state.analyzer.validate_api_key()
            health_info["components"]["analyzer"] = {"status": "healthy", "api_key_valid": api_valid}
        else:
            health_info["components"]["analyzer"] = {"status": "unavailable"}
            
        if hasattr(app.state, 'predictor') and app.state.predictor:
            predictor_info = app.state.predictor.get_model_info()
            health_info["components"]["predictor"] = {"status": "healthy", **predictor_info}
        else:
            health_info["components"]["predictor"] = {"status": "unavailable"}
            
        if hasattr(app.state, 'coaching_system') and app.state.coaching_system:
            health_info["components"]["coaching"] = {"status": "healthy"}
        else:
            health_info["components"]["coaching"] = {"status": "unavailable"}
            
    except Exception as e:
        health_info["components"]["ai"] = {"status": "unhealthy", "error": str(e)}
        health_info["status"] = "degraded"
    
    return health_info


# Call analysis endpoints
@app.post("/analyze", response_model=CallInsight)
async def analyze_call(
    transcript_data: Dict[str, str],
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session_dependency)
):
    """
    Analyze a call transcript and extract insights.
    
    Args:
        transcript_data: Dict containing 'call_id' and 'transcript'
    """
    if not app.state.analyzer:
        raise HTTPException(status_code=503, detail="Analysis service unavailable")
    
    call_id = transcript_data.get("call_id")
    transcript = transcript_data.get("transcript")
    
    if not call_id or not transcript:
        raise HTTPException(status_code=400, detail="call_id and transcript are required")
    
    try:
        # Analyze transcript
        insight = await app.state.analyzer.analyze_transcript(transcript, call_id)
        
        # Save to database in background
        background_tasks.add_task(save_insight_to_db, insight, session)
        
        # Update predictor with new data
        if app.state.predictor:
            background_tasks.add_task(update_predictor_data, [insight])
        
        logger.info("Call analyzed successfully", call_id=call_id, confidence=insight.confidence_score)
        
        return insight
        
    except Exception as e:
        logger.error("Error analyzing call", call_id=call_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/batch")
async def analyze_batch(
    batch_data: List[Dict[str, str]],
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_db_session_dependency)
):
    """
    Analyze multiple call transcripts in batch.
    
    Args:
        batch_data: List of dicts containing 'call_id' and 'transcript'
    """
    if not app.state.analyzer:
        raise HTTPException(status_code=503, detail="Analysis service unavailable")
    
    if len(batch_data) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 calls")
    
    try:
        insights = await app.state.analyzer.analyze_batch(batch_data)
        
        # Save to database in background
        background_tasks.add_task(save_insights_to_db, insights, session)
        
        # Update predictor
        if app.state.predictor:
            background_tasks.add_task(update_predictor_data, insights)
        
        logger.info("Batch analysis completed", batch_size=len(batch_data))
        
        return {"analyzed_count": len(insights), "insights": insights}
        
    except Exception as e:
        logger.error("Error in batch analysis", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")


# Prediction endpoints
@app.get("/predict/week", response_model=PredictionResponse)
async def predict_next_week():
    """Get predictions for the next week."""
    if not app.state.predictor:
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    
    try:
        predictions = app.state.predictor.predict_next_period(days=7)
        logger.info("Weekly predictions generated")
        return predictions
        
    except Exception as e:
        logger.error("Error generating predictions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predict/custom")
async def predict_custom_period(days: int = 7):
    """Get predictions for a custom number of days."""
    if not app.state.predictor:
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    
    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 30")
    
    try:
        predictions = app.state.predictor.predict_next_period(days=days)
        logger.info("Custom predictions generated", days=days)
        return predictions
        
    except Exception as e:
        logger.error("Error generating predictions", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Real-time coaching endpoints
@app.post("/coaching/start")
async def start_coaching_session(
    session_data: Dict[str, Any]
):
    """Start a new real-time coaching session."""
    if not app.state.coaching_system:
        raise HTTPException(status_code=503, detail="Coaching service unavailable")
    
    staff_id = session_data.get("staff_id")
    call_id = session_data.get("call_id")
    customer_context = session_data.get("customer_context")
    
    if not staff_id or not call_id:
        raise HTTPException(status_code=400, detail="staff_id and call_id are required")
    
    try:
        response = await app.state.coaching_system.start_coaching_session(
            staff_id, call_id, customer_context
        )
        logger.info("Coaching session started", staff_id=staff_id, call_id=call_id)
        return response
        
    except Exception as e:
        logger.error("Error starting coaching session", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to start coaching: {str(e)}")


@app.post("/coaching/update", response_model=CoachingResponse)
async def update_coaching_session(
    update_data: Dict[str, Any]
):
    """Update a coaching session with new transcript chunk."""
    if not app.state.coaching_system:
        raise HTTPException(status_code=503, detail="Coaching service unavailable")
    
    call_id = update_data.get("call_id")
    transcript_chunk = update_data.get("transcript_chunk")
    speaker = update_data.get("speaker", "unknown")
    
    if not call_id or not transcript_chunk:
        raise HTTPException(status_code=400, detail="call_id and transcript_chunk are required")
    
    try:
        response = await app.state.coaching_system.process_live_transcript_chunk(
            call_id, transcript_chunk, speaker
        )
        return response
        
    except Exception as e:
        logger.error("Error updating coaching session", call_id=call_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Coaching update failed: {str(e)}")


@app.post("/coaching/end")
async def end_coaching_session(
    end_data: Dict[str, Any]
):
    """End a coaching session."""
    if not app.state.coaching_system:
        raise HTTPException(status_code=503, detail="Coaching service unavailable")
    
    call_id = end_data.get("call_id")
    session_summary = end_data.get("session_summary")
    
    if not call_id:
        raise HTTPException(status_code=400, detail="call_id is required")
    
    try:
        response = await app.state.coaching_system.end_coaching_session(call_id, session_summary)
        logger.info("Coaching session ended", call_id=call_id)
        return response
        
    except Exception as e:
        logger.error("Error ending coaching session", call_id=call_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to end coaching: {str(e)}")


# WebSocket for real-time coaching
@app.websocket("/coaching/ws/{call_id}")
async def coaching_websocket(websocket: WebSocket, call_id: str, staff_id: str):
    """WebSocket endpoint for real-time coaching updates."""
    if not app.state.coaching_system:
        await websocket.close(code=1011, reason="Coaching service unavailable")
        return
    
    await websocket.accept()
    logger.info("Coaching WebSocket connected", call_id=call_id, staff_id=staff_id)
    
    try:
        # Start coaching session
        session_response = await app.state.coaching_system.start_coaching_session(
            staff_id, call_id
        )
        await websocket.send_json({"type": "session_started", "data": session_response})
        
        while True:
            # Receive transcript chunks
            data = await websocket.receive_json()
            
            if data.get("type") == "transcript_chunk":
                transcript_chunk = data.get("transcript_chunk")
                speaker = data.get("speaker", "unknown")
                
                # Process coaching update
                coaching_response = await app.state.coaching_system.process_live_transcript_chunk(
                    call_id, transcript_chunk, speaker
                )
                
                # Send coaching response
                await websocket.send_json({
                    "type": "coaching_update",
                    "data": coaching_response
                })
                
            elif data.get("type") == "end_session":
                # End coaching session
                final_report = await app.state.coaching_system.end_coaching_session(
                    call_id, data.get("session_summary")
                )
                
                await websocket.send_json({
                    "type": "session_ended",
                    "data": final_report
                })
                break
    
    except WebSocketDisconnect:
        logger.info("Coaching WebSocket disconnected", call_id=call_id)
        # Clean up coaching session
        try:
            await app.state.coaching_system.end_coaching_session(call_id)
        except:
            pass
    
    except Exception as e:
        logger.error("WebSocket error", call_id=call_id, error=str(e))
        await websocket.close(code=1011, reason="Internal error")


# Analytics endpoints
@app.get("/analytics/overview", response_model=AnalyticsResponse)
async def get_analytics_overview(
    days: int = 30,
    session: AsyncSession = Depends(get_db_session_dependency)
):
    """Get analytics overview for the specified period."""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query call insights
        stmt = select(CallInsightDB).where(
            CallInsightDB.timestamp >= start_date,
            CallInsightDB.timestamp <= end_date
        )
        result = await session.execute(stmt)
        insights = result.scalars().all()
        
        if not insights:
            return AnalyticsResponse(
                total_calls=0,
                average_sentiment=0.0,
                top_intents=[],
                staff_performance={},
                revenue_opportunities=0.0,
                common_pain_points=[]
            )
        
        # Calculate metrics
        total_calls = len(insights)
        average_sentiment = sum(i.sentiment_score for i in insights) / total_calls
        total_revenue = sum(i.revenue_opportunity for i in insights)
        
        # Top intents
        intent_counts = {}
        for insight in insights:
            intent = insight.primary_intent
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        top_intents = [
            {"intent": intent, "count": count}
            for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Pain points analysis
        all_pain_points = []
        for insight in insights:
            all_pain_points.extend(insight.pain_points or [])
        
        pain_point_counts = {}
        for point in all_pain_points:
            pain_point_counts[point] = pain_point_counts.get(point, 0) + 1
        
        common_pain_points = [
            {"pain_point": point, "count": count}
            for point, count in sorted(pain_point_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]
        
        return AnalyticsResponse(
            total_calls=total_calls,
            average_sentiment=average_sentiment,
            top_intents=top_intents,
            staff_performance={},  # Would need staff performance data
            revenue_opportunities=total_revenue,
            common_pain_points=common_pain_points
        )
        
    except Exception as e:
        logger.error("Error generating analytics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


# Call insights CRUD endpoints
@app.get("/insights")
async def get_call_insights(
    limit: int = 100,
    offset: int = 0,
    session: AsyncSession = Depends(get_db_session_dependency)
):
    """Get paginated call insights."""
    try:
        stmt = select(CallInsightDB).offset(offset).limit(limit).order_by(CallInsightDB.created_at.desc())
        result = await session.execute(stmt)
        insights = result.scalars().all()
        
        return {"insights": insights, "count": len(insights)}
        
    except Exception as e:
        logger.error("Error fetching insights", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch insights")


@app.get("/insights/{call_id}")
async def get_call_insight(
    call_id: str,
    session: AsyncSession = Depends(get_db_session_dependency)
):
    """Get a specific call insight by call_id."""
    try:
        stmt = select(CallInsightDB).where(CallInsightDB.call_id == call_id)
        result = await session.execute(stmt)
        insight = result.scalar_one_or_none()
        
        if not insight:
            raise HTTPException(status_code=404, detail="Call insight not found")
        
        return insight
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching insight", call_id=call_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch insight")


# Model management endpoints
@app.post("/models/train")
async def trigger_model_training():
    """Trigger predictive model training."""
    if not app.state.predictor:
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    
    try:
        training_results = app.state.predictor.train_models()
        logger.info("Model training completed", results=training_results)
        return {"status": "training_completed", "results": training_results}
        
    except Exception as e:
        logger.error("Error training models", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/models/info")
async def get_model_info():
    """Get information about trained models."""
    if not app.state.predictor:
        raise HTTPException(status_code=503, detail="Prediction service unavailable")
    
    try:
        model_info = app.state.predictor.get_model_info()
        return model_info
        
    except Exception as e:
        logger.error("Error getting model info", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get model info")


# Background tasks
async def save_insight_to_db(insight: CallInsight, session: AsyncSession):
    """Save call insight to database."""
    try:
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
        
        session.add(db_insight)
        await session.commit()
        logger.debug("Insight saved to database", call_id=insight.call_id)
        
    except Exception as e:
        await session.rollback()
        logger.error("Error saving insight to database", call_id=insight.call_id, error=str(e))


async def save_insights_to_db(insights: List[CallInsight], session: AsyncSession):
    """Save multiple call insights to database."""
    try:
        db_insights = []
        for insight in insights:
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
            db_insights.append(db_insight)
        
        session.add_all(db_insights)
        await session.commit()
        logger.debug("Batch insights saved to database", count=len(insights))
        
    except Exception as e:
        await session.rollback()
        logger.error("Error saving batch insights to database", error=str(e))


async def update_predictor_data(insights: List[CallInsight]):
    """Update predictor with new insights."""
    try:
        if app.state.predictor:
            app.state.predictor.add_insights(insights)
            logger.debug("Predictor updated with new insights", count=len(insights))
    except Exception as e:
        logger.error("Error updating predictor", error=str(e))


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured logging."""
    logger.warning("HTTP exception", 
                  path=request.url.path,
                  method=request.method,
                  status_code=exc.status_code,
                  detail=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error("Unhandled exception",
                path=request.url.path, 
                method=request.method,
                error=str(exc),
                exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "call_analysis.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers
    )