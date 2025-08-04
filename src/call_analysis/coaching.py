"""
Real-time coaching system for the call analysis platform.

This module provides live coaching capabilities during customer calls,
offering immediate feedback, script suggestions, and escalation alerts
to improve call quality and customer satisfaction.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict, deque

import numpy as np

from .config import get_settings

logger = logging.getLogger(__name__)


class RealTimeCoachingSystem:
    """
    Real-time coaching system for call center staff.
    
    Provides immediate feedback during live calls including:
    - Sentiment monitoring and alerts
    - Script recommendations
    - Performance coaching
    - Escalation triggers
    - Quality assurance monitoring
    """
    
    def __init__(self, coaching_rules_path: Optional[str] = None):
        """
        Initialize the real-time coaching system.
        
        Args:
            coaching_rules_path: Path to custom coaching rules configuration
        """
        settings = get_settings()
        
        # Load coaching configuration
        self.coaching_rules = self._load_coaching_rules(coaching_rules_path)
        
        # Active coaching sessions
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Staff performance baselines and history
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        self.staff_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Quality thresholds for alerts
        self.quality_thresholds = {
            "sentiment_drop": -0.3,
            "sentiment_critical": -0.7,
            "urgency_spike": 4,
            "call_duration_warning": 600,  # 10 minutes
            "call_duration_critical": 900,  # 15 minutes
            "resolution_rate_threshold": 0.8,
            "empathy_threshold": 0.6,
            "professionalism_threshold": 0.7,
        }
        
        # Coaching message templates
        self.coaching_templates = self._initialize_coaching_templates()
        
        # Performance tracking
        self.session_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Real-time analytics
        self.conversation_analytics = ConversationAnalytics()
    
    async def start_coaching_session(self, staff_id: str, call_id: str, 
                                   customer_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Initialize a new real-time coaching session.
        
        Args:
            staff_id: Unique identifier for the staff member
            call_id: Unique identifier for the call
            customer_context: Optional customer information for personalized coaching
            
        Returns:
            Session initialization response with baseline metrics and initial coaching tips
        """
        logger.info(f"Starting coaching session for staff {staff_id}, call {call_id}")
        
        # Initialize session state
        session = {
            "staff_id": staff_id,
            "call_id": call_id,
            "start_time": datetime.now(),
            "customer_context": customer_context or {},
            "transcript_buffer": "",
            "conversation_chunks": [],
            "alerts_triggered": [],
            "coaching_suggestions": [],
            "scripts_recommended": [],
            "escalation_flags": [],
            "performance_metrics": {
                "empathy_score": 0.5,
                "efficiency_score": 0.5,
                "knowledge_score": 0.5,
                "professionalism_score": 0.5,
                "overall_score": 0.5,
            },
            "sentiment_timeline": [],
            "interaction_count": 0,
            "last_coaching_time": datetime.now(),
        }
        
        self.active_sessions[call_id] = session
        
        # Get staff baseline performance
        baseline = self._get_staff_baseline(staff_id)
        
        # Generate initial coaching tips
        initial_tips = self._get_initial_coaching_tips(staff_id, customer_context)
        
        # Set up real-time analytics
        await self.conversation_analytics.initialize_session(call_id, staff_id)
        
        response = {
            "session_id": call_id,
            "status": "active",
            "baseline_performance": baseline,
            "initial_coaching_tips": initial_tips,
            "quality_targets": self.quality_thresholds,
            "customer_insights": self._generate_customer_insights(customer_context),
            "recommended_approach": self._recommend_initial_approach(customer_context),
        }
        
        logger.info(f"Coaching session started for call {call_id}")
        return response
    
    async def process_live_transcript_chunk(self, call_id: str, 
                                         transcript_chunk: str,
                                         speaker: str = "unknown",
                                         timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process a chunk of live transcript and provide real-time coaching.
        
        Args:
            call_id: Call identifier
            transcript_chunk: New transcript text
            speaker: Who is speaking (staff, customer, etc.)
            timestamp: When this chunk occurred
            
        Returns:
            Real-time coaching response with alerts and suggestions
        """
        if call_id not in self.active_sessions:
            logger.warning(f"No active coaching session found for call {call_id}")
            return {"error": "No active coaching session found"}
        
        session = self.active_sessions[call_id]
        timestamp = timestamp or datetime.now()
        
        # Update session with new transcript
        session["transcript_buffer"] += f" {transcript_chunk}"
        session["conversation_chunks"].append({
            "text": transcript_chunk,
            "speaker": speaker,
            "timestamp": timestamp,
        })
        session["interaction_count"] += 1
        
        # Analyze current conversation state
        analysis = await self._analyze_conversation_state(session, transcript_chunk, speaker)
        
        # Generate coaching response
        coaching_response = {
            "timestamp": timestamp.isoformat(),
            "call_id": call_id,
            "interaction_count": session["interaction_count"],
            "immediate_alerts": [],
            "coaching_suggestions": [],
            "performance_updates": {},
            "script_recommendations": [],
            "escalation_flags": [],
            "sentiment_update": analysis.get("current_sentiment", 0.0),
            "conversation_stage": analysis.get("conversation_stage", "unknown"),
            "quality_indicators": analysis.get("quality_indicators", {}),
        }
        
        # Check for immediate alerts
        immediate_alerts = await self._check_immediate_alerts(analysis, session)
        coaching_response["immediate_alerts"] = immediate_alerts
        
        # Generate coaching suggestions  
        coaching_suggestions = await self._generate_coaching_suggestions(analysis, session)
        coaching_response["coaching_suggestions"] = coaching_suggestions
        
        # Recommend script responses
        script_recommendations = await self._suggest_script_responses(analysis, session)
        coaching_response["script_recommendations"] = script_recommendations
        
        # Check escalation triggers
        escalation_flags = await self._check_escalation_triggers(analysis, session)
        coaching_response["escalation_flags"] = escalation_flags
        
        # Update performance metrics
        performance_updates = await self._update_performance_metrics(analysis, session)
        coaching_response["performance_updates"] = performance_updates
        
        # Update session state
        session["alerts_triggered"].extend(immediate_alerts)
        session["coaching_suggestions"].extend(coaching_suggestions)
        session["scripts_recommended"].extend(script_recommendations)
        session["escalation_flags"].extend(escalation_flags)
        session["last_coaching_time"] = timestamp
        
        # Track sentiment timeline
        session["sentiment_timeline"].append({
            "timestamp": timestamp,
            "sentiment": analysis.get("current_sentiment", 0.0),
            "speaker": speaker,
        })
        
        # Update real-time analytics
        await self.conversation_analytics.update_session(call_id, analysis)
        
        return coaching_response
    
    async def end_coaching_session(self, call_id: str, 
                                 session_summary: Optional[Dict] = None) -> Dict[str, Any]:
        """
        End a coaching session and generate final performance report.
        
        Args:
            call_id: Call identifier
            session_summary: Optional final call summary
            
        Returns:
            Final coaching report and performance metrics
        """
        if call_id not in self.active_sessions:
            return {"error": "No active coaching session found"}
        
        session = self.active_sessions[call_id]
        end_time = datetime.now()
        call_duration = (end_time - session["start_time"]).total_seconds()
        
        # Generate final performance report
        final_report = {
            "call_id": call_id,
            "staff_id": session["staff_id"],
            "call_duration": call_duration,
            "start_time": session["start_time"].isoformat(),
            "end_time": end_time.isoformat(),
            "interaction_count": session["interaction_count"],
            "final_performance_metrics": session["performance_metrics"],
            "alerts_summary": self._summarize_alerts(session["alerts_triggered"]),
            "coaching_effectiveness": self._calculate_coaching_effectiveness(session),
            "areas_for_improvement": self._identify_improvement_areas(session),
            "strengths_demonstrated": self._identify_strengths(session),
            "recommended_training": self._recommend_training(session),
            "sentiment_journey": self._analyze_sentiment_journey(session["sentiment_timeline"]),
            "session_summary": session_summary or {},
        }
        
        # Update staff performance history
        self._update_staff_history(session["staff_id"], final_report)
        
        # Store session metrics
        self.session_metrics[call_id] = final_report
        
        # Clean up active session
        del self.active_sessions[call_id]
        
        # Finalize analytics
        await self.conversation_analytics.finalize_session(call_id)
        
        logger.info(f"Coaching session ended for call {call_id}. Duration: {call_duration:.0f}s")
        return final_report
    
    async def _analyze_conversation_state(self, session: Dict, 
                                        transcript_chunk: str, 
                                        speaker: str) -> Dict[str, Any]:
        """
        Analyze current conversation state for coaching insights.
        
        Uses lightweight heuristics for real-time performance while maintaining accuracy.
        """
        analysis = {
            "current_sentiment": self._analyze_sentiment_quick(transcript_chunk),
            "conversation_stage": self._identify_conversation_stage(session["transcript_buffer"]),
            "detected_concerns": self._detect_customer_concerns(transcript_chunk),
            "staff_indicators": self._assess_staff_performance_indicators(transcript_chunk, speaker),
            "customer_state": self._assess_customer_state(transcript_chunk, speaker),
            "urgency_indicators": self._detect_urgency_indicators(transcript_chunk),
            "quality_indicators": self._assess_call_quality(session, transcript_chunk),
        }
        
        return analysis
    
    async def _check_immediate_alerts(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Check for conditions requiring immediate attention.
        """
        alerts = []
        
        # Sentiment deterioration
        current_sentiment = analysis.get("current_sentiment", 0)
        if current_sentiment < self.quality_thresholds["sentiment_critical"]:
            alerts.append({
                "type": "critical",
                "category": "sentiment",
                "message": f"🚨 CRITICAL SENTIMENT ALERT: Customer satisfaction critically low ({current_sentiment:.2f}). Immediate intervention required.",
                "suggested_action": "Consider apologizing, acknowledging concerns, and offering to escalate to supervisor.",
                "priority": "high"
            })
        elif current_sentiment < self.quality_thresholds["sentiment_drop"]:
            alerts.append({
                "type": "warning", 
                "category": "sentiment",
                "message": f"⚠️ SENTIMENT WARNING: Customer satisfaction declining ({current_sentiment:.2f}). Use empathy techniques.",
                "suggested_action": "Acknowledge customer feelings and show understanding.",
                "priority": "medium"
            })
        
        # Call duration alerts
        call_duration = (datetime.now() - session["start_time"]).total_seconds()
        if call_duration > self.quality_thresholds["call_duration_critical"]:
            alerts.append({
                "type": "critical",
                "category": "duration", 
                "message": f"⏰ DURATION CRITICAL: Call exceeding {call_duration/60:.0f} minutes. Consider wrap-up.",
                "suggested_action": "Summarize next steps and schedule follow-up if needed.",
                "priority": "high"
            })
        elif call_duration > self.quality_thresholds["call_duration_warning"]:
            alerts.append({
                "type": "warning",
                "category": "duration",
                "message": f"⏰ DURATION WARNING: Call running long ({call_duration/60:.0f} minutes). Monitor progress.",
                "suggested_action": "Ensure conversation stays focused on resolution.",
                "priority": "medium"
            })
        
        # Customer frustration indicators
        frustration_keywords = ["frustrated", "angry", "upset", "terrible", "awful", "ridiculous"]
        if any(keyword in session["transcript_buffer"].lower() for keyword in frustration_keywords):
            alerts.append({
                "type": "warning",
                "category": "customer_emotion",
                "message": "😤 FRUSTRATION DETECTED: Customer expressing negative emotions.",
                "suggested_action": "Use de-escalation techniques: acknowledge, empathize, and focus on solutions.",
                "priority": "high"
            })
        
        # Emergency/urgent situation indicators
        emergency_keywords = ["emergency", "severe pain", "bleeding", "accident", "urgent", "can't wait"]
        if any(keyword in session["transcript_buffer"].lower() for keyword in emergency_keywords):
            alerts.append({
                "type": "critical",
                "category": "emergency",
                "message": "🚨 EMERGENCY INDICATORS: Urgent medical situation detected.",
                "suggested_action": "Prioritize immediate scheduling. Follow emergency protocols.",
                "priority": "critical"
            })
        
        # Legal/compliance keywords
        legal_keywords = ["lawyer", "legal action", "sue", "complaint", "better business bureau"]
        if any(keyword in session["transcript_buffer"].lower() for keyword in legal_keywords):
            alerts.append({
                "type": "critical",
                "category": "legal",
                "message": "⚖️ LEGAL LANGUAGE DETECTED: Customer mentioned legal action.",
                "suggested_action": "Escalate to management immediately. Document everything carefully.",
                "priority": "critical"
            })
        
        return alerts
    
    async def _generate_coaching_suggestions(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Generate specific, actionable coaching suggestions.
        """
        suggestions = []
        
        # Empathy coaching
        if analysis["current_sentiment"] < 0 and "empathy" not in [s.get("type") for s in session.get("coaching_suggestions", [])]:
            suggestions.append({
                "type": "empathy",
                "message": "💝 EMPATHY OPPORTUNITY: Customer seems concerned. Try acknowledging their feelings.",
                "script_example": "I understand your concern about [specific issue]. Let me help you with that.",
                "timing": "immediate"
            })
        
        # Active listening coaching
        if analysis["conversation_stage"] == "customer_speaking" and session["interaction_count"] % 3 == 0:
            suggestions.append({
                "type": "active_listening",
                "message": "👂 ACTIVE LISTENING: Let customer finish, then summarize what you heard.",
                "script_example": "Let me make sure I understand correctly. You're saying that...",
                "timing": "after_customer_finishes"
            })
        
        # Solution-focused coaching
        if "problem" in analysis.get("detected_concerns", []) and analysis["conversation_stage"] == "problem_solving":
            suggestions.append({
                "type": "solution_focus",
                "message": "🎯 SOLUTION FOCUS: Move toward resolution. Ask what would make this right.",
                "script_example": "What would be the best outcome for you in this situation?",
                "timing": "immediate"
            })
        
        # Closing coaching
        if analysis["conversation_stage"] == "resolution" and session["interaction_count"] > 10:
            suggestions.append({
                "type": "closing",
                "message": "🏁 CLOSING OPPORTUNITY: Summarize agreements and confirm next steps.",
                "script_example": "Let me confirm what we've agreed on today...",
                "timing": "immediate"
            })
        
        # Upselling opportunity (when sentiment is positive)
        if (analysis["current_sentiment"] > 0.3 and 
            any(service in session["transcript_buffer"].lower() for service in ["cleaning", "checkup"]) and
            session["interaction_count"] > 5):
            suggestions.append({
                "type": "upsell",
                "message": "💰 UPSELL OPPORTUNITY: Customer is satisfied. Consider mentioning additional services.",
                "script_example": "While you're here, we could also take a look at...",
                "timing": "when_appropriate"
            })
        
        return suggestions
    
    async def _suggest_script_responses(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Suggest specific script responses based on conversation context.
        """
        scripts = []
        
        concerns = analysis.get("detected_concerns", [])
        customer_state = analysis.get("customer_state", {})
        
        # Pricing concerns
        if "pricing" in concerns:
            scripts.append({
                "category": "pricing",
                "situation": "Customer concerned about cost",
                "script": "I understand cost is important. We offer flexible payment plans, and I can have our treatment coordinator discuss options that work within your budget.",
                "follow_up": "Would you like me to schedule a consultation to go over the options?"
            })
        
        # Availability concerns
        if "availability" in concerns:
            scripts.append({
                "category": "scheduling",
                "situation": "Customer needs appointment soon",
                "script": "Let me check our schedule for the next few weeks. We also maintain a cancellation list if you'd like to be contacted about earlier openings.",
                "follow_up": "What days and times work best for you?"
            })
        
        # Insurance questions
        if "insurance" in concerns:
            scripts.append({
                "category": "insurance",
                "situation": "Insurance coverage questions",
                "script": "We work with most major insurance plans and handle direct billing. I can verify your coverage right now and provide an estimate before your visit.",
                "follow_up": "Can you provide me with your insurance information?"
            })
        
        # Pain/discomfort
        if "pain" in concerns:
            scripts.append({
                "category": "medical",
                "situation": "Customer in pain or discomfort",
                "script": "I'm sorry to hear you're experiencing discomfort. Let's get you scheduled as soon as possible. We have emergency slots available.",
                "follow_up": "Can you describe the type of pain you're experiencing?"
            })
        
        # De-escalation for upset customers
        if customer_state.get("emotion_level", "") == "upset":
            scripts.append({
                "category": "de_escalation",
                "situation": "Customer is upset or frustrated",
                "script": "I sincerely apologize for the frustration this has caused you. Your experience is very important to us, and I want to make this right.",
                "follow_up": "What can I do to resolve this for you today?"
            })
        
        return scripts
    
    async def _check_escalation_triggers(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Identify situations requiring supervisor intervention.
        """
        escalation_flags = []
        
        # Multiple resolution attempts without progress
        transcript = session["transcript_buffer"].lower()
        resolution_attempts = transcript.count("let me") + transcript.count("i'll check") + transcript.count("one moment")
        
        if resolution_attempts > 4 and session["interaction_count"] > 15:
            escalation_flags.append({
                "type": "resolution_difficulty",
                "message": f"ESCALATION SUGGESTED: Multiple resolution attempts ({resolution_attempts}) without clear progress.",
                "reason": "Complex issue requiring supervisor expertise",
                "urgency": "medium"
            })
        
        # Repeated customer frustration
        frustration_indicators = ["frustrated", "angry", "upset", "ridiculous", "unacceptable"]
        frustration_count = sum(transcript.count(word) for word in frustration_indicators)
        
        if frustration_count > 2:
            escalation_flags.append({
                "type": "customer_frustration",
                "message": "ESCALATION REQUIRED: Customer expressing repeated frustration.",
                "reason": "High risk of customer dissatisfaction",
                "urgency": "high"
            })
        
        # Legal or regulatory mentions
        legal_indicators = ["lawyer", "legal", "sue", "complaint", "report", "review"]
        if any(word in transcript for word in legal_indicators):
            escalation_flags.append({
                "type": "legal_risk", 
                "message": "ESCALATION CRITICAL: Legal language detected.",
                "reason": "Potential legal or compliance risk",
                "urgency": "critical"
            })
        
        # Request for supervisor
        supervisor_requests = ["supervisor", "manager", "someone else", "higher up"]
        if any(word in transcript for word in supervisor_requests):
            escalation_flags.append({
                "type": "supervisor_requested",
                "message": "ESCALATION REQUESTED: Customer asking for supervisor.",
                "reason": "Direct customer request",
                "urgency": "high"
            })
        
        return escalation_flags
    
    async def _update_performance_metrics(self, analysis: Dict, session: Dict) -> Dict[str, float]:
        """
        Update real-time performance metrics based on conversation analysis.
        """
        current_metrics = session["performance_metrics"]
        staff_indicators = analysis.get("staff_indicators", {})
        
        # Update empathy score
        if "empathy_shown" in staff_indicators:
            current_metrics["empathy_score"] = (
                current_metrics["empathy_score"] * 0.8 + 
                staff_indicators["empathy_shown"] * 0.2
            )
        
        # Update efficiency score
        if "response_speed" in staff_indicators:
            current_metrics["efficiency_score"] = (
                current_metrics["efficiency_score"] * 0.8 + 
                staff_indicators["response_speed"] * 0.2
            )
        
        # Update knowledge score
        if "accuracy" in staff_indicators:
            current_metrics["knowledge_score"] = (
                current_metrics["knowledge_score"] * 0.8 + 
                staff_indicators["accuracy"] * 0.2
            )
        
        # Update professionalism score
        if "professional_language" in staff_indicators:
            current_metrics["professionalism_score"] = (
                current_metrics["professionalism_score"] * 0.8 + 
                staff_indicators["professional_language"] * 0.2
            )
        
        # Calculate overall score
        current_metrics["overall_score"] = np.mean([
            current_metrics["empathy_score"],
            current_metrics["efficiency_score"], 
            current_metrics["knowledge_score"],
            current_metrics["professionalism_score"]
        ])
        
        return current_metrics
    
    # Helper methods for analysis
    def _analyze_sentiment_quick(self, text: str) -> float:
        """Quick sentiment analysis using keyword matching."""
        positive_words = ["good", "great", "thank", "helpful", "excellent", "satisfied", "appreciate"]
        negative_words = ["bad", "terrible", "awful", "frustrated", "angry", "upset", "disappointed"]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _identify_conversation_stage(self, transcript: str) -> str:
        """Identify current stage of the conversation."""
        transcript_lower = transcript.lower()
        
        if len(transcript) < 50:
            return "opening"
        elif any(word in transcript_lower for word in ["thank", "bye", "goodbye", "have a great"]):
            return "closing"
        elif "?" in transcript[-100:]:
            return "customer_inquiry"
        elif any(word in transcript_lower for word in ["let me", "i'll", "i can"]):
            return "staff_responding"
        else:
            return "discussion"
    
    def _detect_customer_concerns(self, text: str) -> List[str]:
        """Detect specific customer concerns in the text."""
        concerns = []
        text_lower = text.lower()
        
        concern_keywords = {
            "pricing": ["price", "cost", "expensive", "afford", "money", "payment"],
            "availability": ["available", "schedule", "appointment", "book", "when"],
            "insurance": ["insurance", "coverage", "benefits", "copay"],
            "pain": ["pain", "hurt", "ache", "discomfort", "sore"],
            "quality": ["quality", "previous", "last time", "before"],
            "wait_time": ["wait", "long", "delay", "time"],
        }
        
        for concern, keywords in concern_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                concerns.append(concern)
        
        return concerns
    
    def _assess_staff_performance_indicators(self, text: str, speaker: str) -> Dict[str, float]:
        """Assess staff performance indicators from their speech."""
        if speaker.lower() != "staff":
            return {}
        
        text_lower = text.lower()
        indicators = {}
        
        # Empathy indicators
        empathy_words = ["understand", "sorry", "apologize", "feel", "appreciate"]
        indicators["empathy_shown"] = min(1.0, sum(1 for word in empathy_words if word in text_lower) / 3)
        
        # Professional language
        unprofessional = ["um", "uh", "like", "you know", "whatever"]
        professional_score = 1.0 - min(1.0, sum(1 for word in unprofessional if word in text_lower) / 5)
        indicators["professional_language"] = professional_score
        
        # Knowledge indicators
        knowledge_words = ["we offer", "i can help", "let me explain", "the process"]
        indicators["accuracy"] = min(1.0, sum(1 for phrase in knowledge_words if phrase in text_lower) / 2)
        
        # Response completeness
        if len(text.split()) > 10:
            indicators["response_speed"] = 0.8
        else:
            indicators["response_speed"] = 0.5
        
        return indicators
    
    def _assess_customer_state(self, text: str, speaker: str) -> Dict[str, Any]:
        """Assess customer emotional state and engagement."""
        if speaker.lower() != "customer":
            return {}
        
        text_lower = text.lower()
        
        # Emotion level
        high_emotion_words = ["very", "really", "extremely", "absolutely"]
        emotion_level = "high" if any(word in text_lower for word in high_emotion_words) else "normal"
        
        # Satisfaction indicators
        satisfaction_positive = ["good", "great", "thank", "appreciate"]
        satisfaction_negative = ["bad", "terrible", "frustrated", "upset"]
        
        satisfaction = "positive" if any(word in text_lower for word in satisfaction_positive) else \
                      "negative" if any(word in text_lower for word in satisfaction_negative) else "neutral"
        
        return {
            "emotion_level": emotion_level,
            "satisfaction": satisfaction,
            "engagement": "high" if "?" in text else "normal"
        }
    
    def _detect_urgency_indicators(self, text: str) -> int:
        """Detect urgency level from 1-5."""
        text_lower = text.lower()
        
        critical_words = ["emergency", "urgent", "asap", "immediately", "can't wait"]
        high_words = ["soon", "quickly", "right away", "need"]
        
        if any(word in text_lower for word in critical_words):
            return 5
        elif any(word in text_lower for word in high_words):
            return 3
        else:
            return 1
    
    def _assess_call_quality(self, session: Dict, current_text: str) -> Dict[str, float]:
        """Assess overall call quality indicators."""
        return {
            "conversation_flow": 0.8,  # Placeholder - would need more sophisticated analysis
            "issue_resolution_progress": 0.7,
            "customer_engagement": 0.8,
            "staff_effectiveness": session["performance_metrics"]["overall_score"]
        }
    
    # Session management helpers
    def _get_staff_baseline(self, staff_id: str) -> Dict[str, float]:
        """Get baseline performance metrics for staff member."""
        if staff_id in self.performance_baselines:
            return self.performance_baselines[staff_id]
        
        # Default baseline
        return {
            "avg_sentiment": 0.5,
            "avg_resolution_rate": 0.8,
            "avg_call_duration": 300,
            "empathy_score": 0.7,
            "efficiency_score": 0.7,
            "knowledge_score": 0.7,
            "professionalism_score": 0.8
        }
    
    def _get_initial_coaching_tips(self, staff_id: str, customer_context: Optional[Dict]) -> List[str]:
        """Generate personalized initial coaching tips."""
        tips = [
            "Use the customer's name during the conversation to build rapport",
            "Ask open-ended questions to fully understand their needs",
            "Listen actively and acknowledge their concerns before proposing solutions"
        ]
        
        # Add context-specific tips
        if customer_context:
            if customer_context.get("is_new_patient"):
                tips.append("This is a new patient - take extra time to explain our processes")
            if customer_context.get("previous_complaints"):
                tips.append("Customer has previous complaints - be extra attentive and empathetic")
            if customer_context.get("high_value"):
                tips.append("High-value customer - ensure exceptional service and consider additional services")
        
        return tips
    
    def _generate_customer_insights(self, customer_context: Optional[Dict]) -> Dict[str, Any]:
        """Generate insights about the customer for personalized service."""
        if not customer_context:
            return {}
        
        insights = {}
        
        if customer_context.get("visit_history"):
            insights["visit_pattern"] = "Regular patient" if len(customer_context["visit_history"]) > 3 else "Occasional patient"
        
        if customer_context.get("service_preferences"):
            insights["preferred_services"] = customer_context["service_preferences"]
        
        if customer_context.get("communication_style"):
            insights["communication_notes"] = customer_context["communication_style"]
        
        return insights
    
    def _recommend_initial_approach(self, customer_context: Optional[Dict]) -> str:
        """Recommend initial approach based on customer context."""
        if not customer_context:
            return "Use standard professional greeting and ask how you can help today."
        
        if customer_context.get("is_returning_patient"):
            return "Welcome them back and reference their previous visit if appropriate."
        elif customer_context.get("is_new_patient"):
            return "Welcome them warmly and offer to explain our processes."
        elif customer_context.get("previous_complaints"):
            return "Acknowledge any previous issues and assure them of your commitment to excellent service."
        else:
            return "Use warm, professional greeting and focus on understanding their needs."
    
    # Utility methods for coaching rules and templates
    def _load_coaching_rules(self, rules_path: Optional[str]) -> Dict[str, Any]:
        """Load coaching rules from configuration."""
        # Default rules - in production, this would load from file
        return {
            "sentiment_monitoring": {
                "enabled": True,
                "check_frequency": 30,  # seconds
                "alert_threshold": -0.3
            },
            "performance_tracking": {
                "enabled": True,
                "metrics": ["empathy", "efficiency", "knowledge", "professionalism"]
            },
            "escalation_rules": {
                "auto_escalate_keywords": ["lawyer", "sue", "complaint"],
                "escalate_after_attempts": 5,
                "escalate_duration_minutes": 15
            }
        }
    
    def _initialize_coaching_templates(self) -> Dict[str, List[str]]:
        """Initialize coaching message templates."""
        return {
            "empathy": [
                "I understand how you feel about this situation.",
                "That must be frustrating for you.",
                "I can see why that would be concerning."
            ],
            "reassurance": [
                "We'll take care of this for you right away.",
                "You're in good hands with our team.",
                "We have extensive experience with this type of situation."
            ],
            "solution_focus": [
                "Let me work on finding a solution for you.",
                "What would be the best outcome for you?",
                "I want to make sure we resolve this to your satisfaction."
            ]
        }
    
    # Reporting and analytics methods
    def _summarize_alerts(self, alerts: List[Dict]) -> Dict[str, Any]:
        """Summarize alerts triggered during the session."""
        alert_counts = defaultdict(int)
        for alert in alerts:
            alert_counts[alert.get("category", "unknown")] += 1
        
        return {
            "total_alerts": len(alerts),
            "by_category": dict(alert_counts),
            "critical_alerts": len([a for a in alerts if a.get("priority") == "critical"])
        }
    
    def _calculate_coaching_effectiveness(self, session: Dict) -> Dict[str, float]:
        """Calculate how effective the coaching was during the session."""
        # Analyze sentiment improvement over time
        sentiment_timeline = session.get("sentiment_timeline", [])
        if len(sentiment_timeline) > 1:
            initial_sentiment = sentiment_timeline[0]["sentiment"]
            final_sentiment = sentiment_timeline[-1]["sentiment"]
            sentiment_improvement = final_sentiment - initial_sentiment
        else:
            sentiment_improvement = 0.0
        
        # Performance metric improvement
        initial_performance = 0.5  # Baseline
        final_performance = session["performance_metrics"]["overall_score"]
        performance_improvement = final_performance - initial_performance
        
        return {
            "sentiment_improvement": sentiment_improvement,
            "performance_improvement": performance_improvement,
            "coaching_interventions": len(session.get("coaching_suggestions", [])),
            "alerts_resolved": 0.8  # Placeholder - would need to track resolution
        }
    
    def _identify_improvement_areas(self, session: Dict) -> List[str]:
        """Identify areas where the staff member can improve."""
        areas = []
        metrics = session["performance_metrics"]
        
        if metrics["empathy_score"] < 0.7:
            areas.append("Empathy and emotional connection with customers")
        if metrics["efficiency_score"] < 0.7:
            areas.append("Call efficiency and time management")
        if metrics["knowledge_score"] < 0.7:
            areas.append("Product and service knowledge")
        if metrics["professionalism_score"] < 0.7:
            areas.append("Professional communication and language")
        
        # Alert-based improvements
        alerts = session.get("alerts_triggered", [])
        if any(alert.get("category") == "sentiment" for alert in alerts):
            areas.append("Customer satisfaction and relationship management")
        
        return areas
    
    def _identify_strengths(self, session: Dict) -> List[str]:
        """Identify demonstrated strengths during the call."""
        strengths = []
        metrics = session["performance_metrics"]
        
        if metrics["empathy_score"] > 0.8:
            strengths.append("Excellent empathy and customer connection")
        if metrics["efficiency_score"] > 0.8:
            strengths.append("Efficient call handling and time management")
        if metrics["knowledge_score"] > 0.8:
            strengths.append("Strong product and service knowledge")
        if metrics["professionalism_score"] > 0.8:
            strengths.append("Professional communication style")
        
        return strengths
    
    def _recommend_training(self, session: Dict) -> List[str]:
        """Recommend specific training based on session performance."""
        training_recommendations = []
        improvement_areas = self._identify_improvement_areas(session)
        
        training_map = {
            "Empathy and emotional connection with customers": "Customer Service Excellence Training",
            "Call efficiency and time management": "Efficient Call Handling Workshop",
            "Product and service knowledge": "Product Knowledge Refresher Course",
            "Professional communication and language": "Professional Communication Skills Training",
            "Customer satisfaction and relationship management": "Customer Relationship Management Training"
        }
        
        for area in improvement_areas:
            if area in training_map:
                training_recommendations.append(training_map[area])
        
        return training_recommendations
    
    def _analyze_sentiment_journey(self, sentiment_timeline: List[Dict]) -> Dict[str, Any]:
        """Analyze how sentiment changed throughout the call."""
        if not sentiment_timeline:
            return {}
        
        sentiments = [s["sentiment"] for s in sentiment_timeline]
        
        return {
            "initial_sentiment": sentiments[0] if sentiments else 0,
            "final_sentiment": sentiments[-1] if sentiments else 0,
            "sentiment_change": sentiments[-1] - sentiments[0] if len(sentiments) > 1 else 0,
            "lowest_point": min(sentiments) if sentiments else 0,
            "highest_point": max(sentiments) if sentiments else 0,
            "volatility": np.std(sentiments) if len(sentiments) > 1 else 0
        }
    
    def _update_staff_history(self, staff_id: str, session_report: Dict) -> None:
        """Update staff performance history with session results."""
        self.staff_history[staff_id].append({
            "call_id": session_report["call_id"],
            "date": session_report["end_time"],
            "performance_metrics": session_report["final_performance_metrics"],
            "call_duration": session_report["call_duration"],
            "alerts_count": session_report["alerts_summary"]["total_alerts"],
            "coaching_effectiveness": session_report["coaching_effectiveness"]
        })
        
        # Update baseline performance
        recent_sessions = list(self.staff_history[staff_id])[-10:]  # Last 10 sessions
        if recent_sessions:
            avg_metrics = {}
            for metric in ["empathy_score", "efficiency_score", "knowledge_score", "professionalism_score"]:
                values = [s["performance_metrics"][metric] for s in recent_sessions if metric in s["performance_metrics"]]
                avg_metrics[metric] = np.mean(values) if values else 0.5
            
            self.performance_baselines[staff_id] = avg_metrics


class ConversationAnalytics:
    """Helper class for real-time conversation analytics."""
    
    def __init__(self):
        self.session_analytics = {}
    
    async def initialize_session(self, call_id: str, staff_id: str):
        """Initialize analytics for a new session."""
        self.session_analytics[call_id] = {
            "staff_id": staff_id,
            "start_time": datetime.now(),
            "interaction_patterns": [],
            "sentiment_history": [],
            "topic_tracking": defaultdict(int)
        }
    
    async def update_session(self, call_id: str, analysis: Dict):
        """Update session analytics with new analysis."""
        if call_id not in self.session_analytics:
            return
        
        analytics = self.session_analytics[call_id]
        
        # Track sentiment history
        analytics["sentiment_history"].append({
            "timestamp": datetime.now(),
            "sentiment": analysis.get("current_sentiment", 0)
        })
        
        # Track conversation topics
        for concern in analysis.get("detected_concerns", []):
            analytics["topic_tracking"][concern] += 1
    
    async def finalize_session(self, call_id: str):
        """Finalize analytics for completed session."""
        if call_id in self.session_analytics:
            # Could save analytics to database or generate reports
            del self.session_analytics[call_id]


# Factory function
def create_coaching_system(coaching_rules_path: Optional[str] = None) -> RealTimeCoachingSystem:
    """
    Create a configured RealTimeCoachingSystem instance.
    
    Args:
        coaching_rules_path: Path to coaching rules configuration
        
    Returns:
        Configured coaching system instance
    """
    return RealTimeCoachingSystem(coaching_rules_path=coaching_rules_path)