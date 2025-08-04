"""
Analysis Orchestrator Agent for coordinating call analysis workflows.

This agent orchestrates the complete analysis pipeline including
transcription, sentiment analysis, entity extraction, topic modeling,
and insight generation for incoming calls.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .base_agent import BaseAgent, AgentStatus, AgentCapability, MessageType
from ..config import get_settings
from ..models import CallInsight, CallData
from ..nlp.sentiment_analyzer import SentimentAnalyzer
from ..nlp.entity_extraction import EntityExtractor
from ..nlp.topic_modeling import TopicModelingEngine
from ..nlp.intent_detection import IntentClassifier

logger = logging.getLogger(__name__)


class AnalysisOrchestrator(BaseAgent):
    """
    Orchestrates complete call analysis workflows.
    
    Coordinates multiple analysis engines to provide comprehensive
    insights including sentiment, entities, topics, and intents.
    """
    
    def __init__(
        self,
        agent_id: str = "analysis_orchestrator",
        name: str = "Analysis Orchestrator"
    ):
        super().__init__(
            agent_id=agent_id,
            name=name,
            description="Orchestrates comprehensive call analysis workflows",
            max_concurrent_tasks=15
        )
        
        self.settings = get_settings()
        
        # Analysis engines
        self.sentiment_analyzer = None
        self.entity_extractor = None
        self.topic_engine = None
        self.intent_classifier = None
        
        # Analysis queue and results
        self.analysis_queue = asyncio.Queue()
        self.analysis_results: Dict[str, Dict[str, Any]] = {}
        self.failed_analyses: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_analysis_time": 0.0,
            "last_analysis": None
        }
        
        # Setup capabilities
        self._setup_capabilities()
        
        # Register message handlers
        self.register_handler("new_call", self._handle_new_call)
        self.register_handler("analyze_call", self._handle_analyze_call)
        self.register_handler("get_analysis", self._handle_get_analysis)
        self.register_handler("batch_analyze", self._handle_batch_analyze)
        self.register_handler("analysis_stats", self._handle_analysis_stats)
    
    def _setup_capabilities(self):
        """Setup orchestrator capabilities."""
        capabilities = [
            AgentCapability(
                name="comprehensive_analysis",
                description="Complete call analysis including sentiment, entities, topics, and intents",
                input_types=["call_transcript", "call_metadata"],
                output_types=["analysis_results", "call_insights"],
                estimated_duration=timedelta(seconds=45)
            ),
            AgentCapability(
                name="sentiment_analysis",
                description="Analyze emotional tone and sentiment in calls",
                input_types=["text"],
                output_types=["sentiment_scores", "emotion_classification"],
                estimated_duration=timedelta(seconds=5)
            ),
            AgentCapability(
                name="entity_extraction",
                description="Extract named entities and key information",
                input_types=["text"],
                output_types=["entities", "structured_data"],
                estimated_duration=timedelta(seconds=10)
            ),
            AgentCapability(
                name="topic_modeling",
                description="Identify main topics and themes in conversations",
                input_types=["text"],
                output_types=["topics", "theme_analysis"],
                estimated_duration=timedelta(seconds=15)
            ),
            AgentCapability(
                name="intent_classification",
                description="Classify customer intents and purposes",
                input_types=["text", "conversation_context"],
                output_types=["intent_labels", "confidence_scores"],
                estimated_duration=timedelta(seconds=8)
            ),
            AgentCapability(
                name="insight_generation",
                description="Generate actionable insights from analysis results",
                input_types=["analysis_results"],
                output_types=["insights", "recommendations"],
                estimated_duration=timedelta(seconds=20)
            )
        ]
        
        for capability in capabilities:
            self.add_capability(capability)
    
    async def initialize(self) -> None:
        """Initialize the analysis orchestrator."""
        logger.info("Initializing Analysis Orchestrator")
        
        # Initialize analysis engines
        try:
            self.sentiment_analyzer = SentimentAnalyzer()
            self.entity_extractor = EntityExtractor()
            self.topic_engine = TopicModelingEngine()
            self.intent_classifier = IntentClassifier()
            
            logger.info("Analysis engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing analysis engines: {e}")
            self.status = AgentStatus.ERROR
            raise
        
        # Start background processors
        self._background_tasks.extend([
            asyncio.create_task(self._analysis_processor()),
            asyncio.create_task(self._results_manager()),
            asyncio.create_task(self._stats_updater())
        ])
        
        logger.info("Analysis Orchestrator initialized successfully")
    
    async def cleanup(self) -> None:
        """Cleanup orchestrator resources."""
        logger.info("Cleaning up Analysis Orchestrator")
        
        # Process remaining analyses
        while not self.analysis_queue.empty():
            try:
                analysis_task = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)
                await self._process_analysis(analysis_task)
            except asyncio.TimeoutError:
                break
        
        # Save analysis results if needed
        await self._save_analysis_results()
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process orchestrator tasks."""
        task_type = task.get("type")
        task_data = task.get("data", {})
        
        if task_type == "analyze_call":
            return await self._analyze_single_call(task_data)
        elif task_type == "batch_analyze":
            return await self._batch_analyze_calls(task_data)
        elif task_type == "generate_insights":
            return await self._generate_insights(task_data)
        elif task_type == "health_check":
            return await self._perform_health_check()
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def analyze_call(
        self,
        call_id: str,
        transcript: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: int = 2
    ) -> str:
        """
        Submit a call for comprehensive analysis.
        
        Args:
            call_id: Unique call identifier
            transcript: Call transcript text
            metadata: Additional call metadata
            priority: Analysis priority
            
        Returns:
            Analysis task ID
        """
        analysis_task = {
            "call_id": call_id,
            "transcript": transcript,
            "metadata": metadata or {},
            "priority": priority,
            "submitted_at": datetime.now(),
            "status": "submitted"
        }
        
        await self.analysis_queue.put(analysis_task)
        
        logger.info(f"Call {call_id} submitted for analysis")
        return f"analysis_{call_id}_{datetime.now().timestamp()}"
    
    async def _analyze_single_call(self, call_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single call comprehensively."""
        call_id = call_data.get("call_id", "unknown")
        transcript = call_data.get("transcript", "")
        metadata = call_data.get("metadata", {})
        
        start_time = datetime.now()
        analysis_results = {}
        
        try:
            # Parallel analysis execution
            analysis_tasks = [
                self._run_sentiment_analysis(transcript),
                self._run_entity_extraction(transcript),
                self._run_topic_modeling(transcript),
                self._run_intent_classification(transcript, metadata)
            ]
            
            sentiment_result, entities_result, topics_result, intent_result = await asyncio.gather(
                *analysis_tasks, return_exceptions=True
            )
            
            # Compile results
            analysis_results = {
                "call_id": call_id,
                "timestamp": start_time.isoformat(),
                "sentiment": sentiment_result if not isinstance(sentiment_result, Exception) else None,
                "entities": entities_result if not isinstance(entities_result, Exception) else None,
                "topics": topics_result if not isinstance(topics_result, Exception) else None,
                "intent": intent_result if not isinstance(intent_result, Exception) else None,
                "metadata": metadata
            }
            
            # Generate insights
            insights = await self._generate_call_insights(analysis_results)
            analysis_results["insights"] = insights
            
            # Calculate analysis duration
            duration = (datetime.now() - start_time).total_seconds()
            analysis_results["analysis_duration"] = duration
            
            # Store results
            self.analysis_results[call_id] = analysis_results
            
            # Update statistics
            self.stats["successful_analyses"] += 1
            self._update_average_analysis_time(duration)
            
            logger.info(f"Successfully analyzed call {call_id} in {duration:.2f}s")
            
            # Notify completion
            await self.send_message(
                "monitoring_agent",
                "analysis_completed",
                {
                    "call_id": call_id,
                    "analysis_results": analysis_results,
                    "duration": duration
                },
                MessageType.NOTIFICATION,
                priority=2
            )
            
            return {
                "status": "success",
                "call_id": call_id,
                "analysis_results": analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error analyzing call {call_id}: {e}")
            
            # Store failed analysis
            self.failed_analyses.append({
                "call_id": call_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "transcript": transcript[:200] + "..." if len(transcript) > 200 else transcript
            })
            
            self.stats["failed_analyses"] += 1
            
            return {
                "status": "error",
                "call_id": call_id,
                "error": str(e)
            }
    
    async def _run_sentiment_analysis(self, transcript: str) -> Dict[str, Any]:
        """Run sentiment analysis on transcript."""
        try:
            if self.sentiment_analyzer:
                return self.sentiment_analyzer.analyze_sentiment(transcript)
            else:
                return {"error": "Sentiment analyzer not available"}
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {"error": str(e)}
    
    async def _run_entity_extraction(self, transcript: str) -> Dict[str, Any]:
        """Run entity extraction on transcript."""
        try:
            if self.entity_extractor:
                return self.entity_extractor.extract_all(transcript)
            else:
                return {"error": "Entity extractor not available"}
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return {"error": str(e)}
    
    async def _run_topic_modeling(self, transcript: str) -> Dict[str, Any]:
        """Run topic modeling on transcript."""
        try:
            if self.topic_engine:
                return self.topic_engine.analyze_topics(transcript)
            else:
                return {"error": "Topic engine not available"}
        except Exception as e:
            logger.error(f"Topic modeling error: {e}")
            return {"error": str(e)}
    
    async def _run_intent_classification(self, transcript: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run intent classification on transcript."""
        try:
            if self.intent_classifier:
                return self.intent_classifier.classify_intent(transcript, metadata)
            else:
                return {"error": "Intent classifier not available"}
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return {"error": str(e)}
    
    async def _generate_call_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable insights from analysis results."""
        insights = {
            "key_insights": [],
            "recommendations": [],
            "risk_factors": [],
            "opportunities": [],
            "follow_up_actions": []
        }
        
        try:
            sentiment = analysis_results.get("sentiment")
            entities = analysis_results.get("entities")
            topics = analysis_results.get("topics")
            intent = analysis_results.get("intent")
            
            # Sentiment-based insights
            if sentiment and not sentiment.get("error"):
                sentiment_score = sentiment.get("compound_score", 0)
                if sentiment_score < -0.5:
                    insights["risk_factors"].append("Highly negative customer sentiment detected")
                    insights["follow_up_actions"].append("Consider immediate follow-up call")
                elif sentiment_score > 0.5:
                    insights["opportunities"].append("Positive customer interaction - potential for upselling")
            
            # Entity-based insights
            if entities and not entities.get("error"):
                if entities.get("phone_numbers"):
                    insights["key_insights"].append("Customer contact information captured")
                
                medical_terms = entities.get("medical_terms", [])
                if medical_terms:
                    insights["key_insights"].append(f"Medical concerns mentioned: {', '.join(medical_terms[:3])}")
                
                if entities.get("insurance_providers"):
                    insights["key_insights"].append("Insurance information discussed")
            
            # Topic-based insights
            if topics and not topics.get("error"):
                dominant_topic = topics.get("dominant_topic", "")
                if "emergency" in dominant_topic.lower():
                    insights["risk_factors"].append("Emergency dental situation identified")
                    insights["follow_up_actions"].append("Prioritize scheduling urgently")
                elif "appointment" in dominant_topic.lower():
                    insights["opportunities"].append("Scheduling opportunity identified")
            
            # Intent-based insights
            if intent and not intent.get("error"):
                primary_intent = intent.get("primary_intent", "")
                if primary_intent == "complaint":
                    insights["risk_factors"].append("Customer complaint detected")
                    insights["follow_up_actions"].append("Escalate to management")
                elif primary_intent == "booking":
                    insights["opportunities"].append("Booking intent identified")
                    insights["recommendations"].append("Provide available appointment slots")
            
            # Generate summary insight
            if insights["risk_factors"]:
                insights["summary"] = "Call requires attention due to identified risk factors"
                insights["priority"] = "high"
            elif insights["opportunities"]:
                insights["summary"] = "Positive interaction with business opportunities"
                insights["priority"] = "medium"
            else:
                insights["summary"] = "Standard customer interaction"
                insights["priority"] = "low"
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights["error"] = str(e)
        
        return insights
    
    async def _analysis_processor(self) -> None:
        """Process queued analysis tasks."""
        while self._running:
            try:
                # Get analysis task from queue
                analysis_task = await asyncio.wait_for(self.analysis_queue.get(), timeout=1.0)
                
                # Submit for processing
                await self.submit_task("analyze_call", analysis_task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in analysis processor: {e}")
    
    async def _results_manager(self) -> None:
        """Manage analysis results and cleanup."""
        while self._running:
            try:
                # Clean up old results (keep last 1000)
                if len(self.analysis_results) > 1000:
                    # Keep most recent results
                    sorted_results = sorted(
                        self.analysis_results.items(),
                        key=lambda x: x[1].get("timestamp", ""),
                        reverse=True
                    )
                    self.analysis_results = dict(sorted_results[:500])
                
                # Clean up old failed analyses (keep last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.failed_analyses = [
                    failure for failure in self.failed_analyses
                    if datetime.fromisoformat(failure["timestamp"]) > cutoff_time
                ]
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in results manager: {e}")
                await asyncio.sleep(60)
    
    async def _stats_updater(self) -> None:
        """Update analysis statistics."""
        while self._running:
            try:
                self.stats["total_analyses"] = (
                    self.stats["successful_analyses"] + 
                    self.stats["failed_analyses"]
                )
                
                if self.stats["successful_analyses"] > 0:
                    self.stats["last_analysis"] = datetime.now().isoformat()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
                await asyncio.sleep(30)
    
    def _update_average_analysis_time(self, duration: float) -> None:
        """Update average analysis time."""
        total_analyses = self.stats["successful_analyses"]
        if total_analyses == 1:
            self.stats["average_analysis_time"] = duration
        else:
            current_avg = self.stats["average_analysis_time"]
            self.stats["average_analysis_time"] = (
                (current_avg * (total_analyses - 1) + duration) / total_analyses
            )
    
    async def _save_analysis_results(self) -> None:
        """Save analysis results to disk."""
        try:
            results_file = self.settings.data_dir / "analysis_results.json"
            
            # Save recent results
            recent_results = dict(list(self.analysis_results.items())[-100:])
            
            async with asyncio.to_thread(open, results_file, 'w') as f:
                json.dump(recent_results, f, indent=2, default=str)
            
            logger.info(f"Saved {len(recent_results)} analysis results")
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform health check on analysis engines."""
        health_status = {
            "overall_status": "healthy",
            "engines": {},
            "issues": []
        }
        
        # Check each analysis engine
        engines = {
            "sentiment_analyzer": self.sentiment_analyzer,
            "entity_extractor": self.entity_extractor,
            "topic_engine": self.topic_engine,
            "intent_classifier": self.intent_classifier
        }
        
        for engine_name, engine in engines.items():
            if engine is None:
                health_status["engines"][engine_name] = "unavailable"
                health_status["issues"].append(f"{engine_name} not initialized")
            else:
                health_status["engines"][engine_name] = "available"
        
        # Check queue status
        queue_size = self.analysis_queue.qsize()
        if queue_size > 50:
            health_status["issues"].append(f"Analysis queue backlog: {queue_size} items")
        
        # Update overall status
        if health_status["issues"]:
            health_status["overall_status"] = "degraded" if len(health_status["issues"]) < 3 else "unhealthy"
        
        health_status["statistics"] = self.stats
        
        return health_status
    
    # Message handlers
    
    async def _handle_new_call(self, message) -> None:
        """Handle new call notification."""
        try:
            payload = message.payload
            call_id = payload.get("call_id")
            call_data = payload.get("call_data", {})
            
            transcript = call_data.get("transcript", "")
            if transcript:
                await self.analyze_call(call_id, transcript, call_data)
            
        except Exception as e:
            logger.error(f"Error handling new call: {e}")
    
    async def _handle_analyze_call(self, message) -> None:
        """Handle explicit analysis request."""
        try:
            payload = message.payload
            result = await self._analyze_single_call(payload)
            
            await self.send_message(
                message.sender,
                "analysis_result",
                result,
                MessageType.RESPONSE
            )
            
        except Exception as e:
            await self.send_message(
                message.sender,
                "analysis_error",
                {"error": str(e)},
                MessageType.ERROR
            )
    
    async def _handle_get_analysis(self, message) -> None:
        """Handle request for analysis results."""
        call_id = message.payload.get("call_id")
        
        if call_id in self.analysis_results:
            result = self.analysis_results[call_id]
        else:
            result = {"error": f"No analysis found for call {call_id}"}
        
        await self.send_message(
            message.sender,
            "analysis_result",
            result,
            MessageType.RESPONSE
        )
    
    async def _handle_batch_analyze(self, message) -> None:
        """Handle batch analysis request."""
        try:
            calls = message.payload.get("calls", [])
            results = []
            
            for call_data in calls:
                result = await self._analyze_single_call(call_data)
                results.append(result)
            
            await self.send_message(
                message.sender,
                "batch_analysis_result",
                {"results": results},
                MessageType.RESPONSE
            )
            
        except Exception as e:
            await self.send_message(
                message.sender,
                "batch_analysis_error",
                {"error": str(e)},
                MessageType.ERROR
            )
    
    async def _handle_analysis_stats(self, message) -> None:
        """Handle request for analysis statistics."""
        await self.send_message(
            message.sender,
            "analysis_stats_response",
            self.stats,
            MessageType.RESPONSE
        )