import os
import re
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# AI/ML imports
try:
    from openai import AsyncOpenAI
    import tiktoken
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    import joblib
    HAS_AI_LIBS = True
except ImportError:
    HAS_AI_LIBS = False
    print("AI libraries not installed. Install with: pip install openai scikit-learn tiktoken")

# Mental model: This is our command center for dental call intelligence.
# Think of it as upgrading from a basic calculator to having a team of
# expert analysts, predictive modelers, and real-time coaches all working
# together to optimize your business operations.

@dataclass
class CallInsight:
    """
    Structured representation of AI-extracted insights from a single call.
    Much richer than simple keyword counting - this captures the full
    context and business implications of each conversation.
    """
    call_id: str
    timestamp: datetime
    clinic_mentioned: str
    primary_intent: str
    secondary_intents: List[str]
    sentiment_score: float  # -1 to 1
    emotional_journey: Dict[str, float]  # beginning, middle, end emotions
    urgency_level: int  # 1-5 scale
    resolution_status: str  # resolved, escalated, follow_up_needed
    staff_performance: Dict[str, float]
    revenue_opportunity: float  # estimated dollar value
    service_requests: List[str]
    pain_points: List[str]
    success_factors: List[str]
    next_actions: List[str]
    confidence_score: float  # AI confidence in analysis

class SemanticTranscriptAnalyzer:
    """
    Phase 1: Replace regex matching with LLM semantic analysis
    
    Mental model: Instead of hunting for exact phrases like a word search puzzle,
    we're teaching an AI to understand the meaning behind what people are saying.
    Think of it as upgrading from a simple calculator to having a smart assistant
    who's listened to thousands of dental calls and understands the nuances.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        if not HAS_AI_LIBS:
            raise ImportError("AI libraries required. Run: pip install openai scikit-learn tiktoken")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Cache for embeddings and analysis to avoid redundant API calls
        self.analysis_cache = {}
        
        # Define our semantic understanding framework
        self.analysis_prompt_template = """
        You are an expert dental practice analyst. Analyze this call transcript and extract structured insights.
        
        Transcript: {transcript}
        
        Extract the following information as valid JSON:
        {{
            "primary_intent": "appointment_booking|emergency|inquiry|complaint|cancellation|insurance|pricing|services",
            "secondary_intents": ["list", "of", "additional", "intents"],
            "sentiment_score": -1.0 to 1.0,
            "emotional_journey": {{
                "beginning": -1.0 to 1.0,
                "middle": -1.0 to 1.0, 
                "end": -1.0 to 1.0
            }},
            "urgency_level": 1 to 5,
            "resolution_status": "resolved|escalated|follow_up_needed|unresolved",
            "staff_performance": {{
                "empathy": 0.0 to 1.0,
                "efficiency": 0.0 to 1.0,
                "knowledge": 0.0 to 1.0,
                "professionalism": 0.0 to 1.0
            }},
            "revenue_opportunity": estimated_dollar_value,
            "service_requests": ["cleaning", "filling", "etc"],
            "pain_points": ["wait_time", "pricing", "availability", "etc"],
            "success_factors": ["friendly_staff", "quick_response", "etc"],
            "next_actions": ["schedule_followup", "send_pricing", "etc"],
            "confidence_score": 0.0 to 1.0,
            "clinic_mentioned": "clinic_name_if_mentioned"
        }}
        
        Be precise and evidence-based in your analysis.
        """
    
    async def analyze_transcript(self, transcript: str, call_id: str) -> CallInsight:
        """
        Core semantic analysis function. This is where the magic happens -
        we're asking an AI to be our expert call center analyst who can
        understand context, emotion, and business implications.
        """
        
        # Check cache first to avoid redundant API calls
        cache_key = f"{call_id}_{hash(transcript[:100])}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        
        try:
            # Chunk transcript if too long for context window
            max_tokens = 3000  # Leave room for prompt and response
            transcript_tokens = len(self.encoding.encode(transcript))
            
            if transcript_tokens > max_tokens:
                # Take first and last portions to capture full conversation arc
                transcript = self._intelligent_chunking(transcript, max_tokens)
            
            prompt = self.analysis_prompt_template.format(transcript=transcript)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert dental practice analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent analysis
                max_tokens=1000
            )
            
            # Parse AI response
            analysis_json = json.loads(response.choices[0].message.content)
            
            # Create structured insight object
            insight = CallInsight(
                call_id=call_id,
                timestamp=datetime.now(),
                clinic_mentioned=analysis_json.get("clinic_mentioned", "unknown"),
                primary_intent=analysis_json.get("primary_intent", "unknown"),
                secondary_intents=analysis_json.get("secondary_intents", []),
                sentiment_score=analysis_json.get("sentiment_score", 0.0),
                emotional_journey=analysis_json.get("emotional_journey", {}),
                urgency_level=analysis_json.get("urgency_level", 1),
                resolution_status=analysis_json.get("resolution_status", "unknown"),
                staff_performance=analysis_json.get("staff_performance", {}),
                revenue_opportunity=analysis_json.get("revenue_opportunity", 0.0),
                service_requests=analysis_json.get("service_requests", []),
                pain_points=analysis_json.get("pain_points", []),
                success_factors=analysis_json.get("success_factors", []),
                next_actions=analysis_json.get("next_actions", []),
                confidence_score=analysis_json.get("confidence_score", 0.0)
            )
            
            # Cache the result
            self.analysis_cache[cache_key] = insight
            return insight
            
        except Exception as e:
            logging.error(f"Error analyzing transcript {call_id}: {e}")
            # Return fallback insight
            return self._create_fallback_insight(call_id, transcript)
    
    def _intelligent_chunking(self, transcript: str, max_tokens: int) -> str:
        """
        Smart chunking that preserves conversation flow and key information.
        We want the beginning (context setting) and end (resolution) of calls.
        """
        
        tokens = self.encoding.encode(transcript)
        if len(tokens) <= max_tokens:
            return transcript
        
        # Take first 40% and last 40% of the conversation
        # This captures initial context and final resolution
        chunk_size = max_tokens // 2
        first_chunk = self.encoding.decode(tokens[:chunk_size])
        last_chunk = self.encoding.decode(tokens[-chunk_size:])
        
        return f"{first_chunk}\n\n[... middle of conversation omitted ...]\n\n{last_chunk}"
    
    def _create_fallback_insight(self, call_id: str, transcript: str) -> CallInsight:
        """
        Fallback analysis when AI fails - better than crashing.
        Uses simple heuristics as backup.
        """
        
        # Basic sentiment from simple word counting
        positive_words = ["good", "great", "thank", "helpful", "satisfied"]
        negative_words = ["bad", "terrible", "rude", "wait", "frustrated"]
        
        pos_count = sum(1 for word in positive_words if word in transcript.lower())
        neg_count = sum(1 for word in negative_words if word in transcript.lower())
        
        sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        
        return CallInsight(
            call_id=call_id,
            timestamp=datetime.now(),
            clinic_mentioned="unknown",
            primary_intent="inquiry",
            secondary_intents=[],
            sentiment_score=sentiment,
            emotional_journey={"beginning": 0.0, "middle": 0.0, "end": 0.0},
            urgency_level=1,
            resolution_status="unknown",
            staff_performance={"empathy": 0.5, "efficiency": 0.5, "knowledge": 0.5, "professionalism": 0.5},
            revenue_opportunity=100.0,  # Conservative estimate
            service_requests=[],
            pain_points=[],
            success_factors=[],
            next_actions=[],
            confidence_score=0.1  # Low confidence for fallback
        )

class PredictiveAnalytics:
    """
    Phase 2: Predictive analytics for operational planning
    
    Mental model: We're building a weather forecast system for your business.
    Instead of predicting rain, we're predicting call volume spikes,
    seasonal service demands, and staffing needs. This transforms reactive
    management into proactive optimization.
    """
    
    def __init__(self, historical_data_path: Optional[str] = None):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Load historical data if available
        if historical_data_path and os.path.exists(historical_data_path):
            self.historical_insights = self._load_historical_data(historical_data_path)
        else:
            self.historical_insights = []
    
    def add_insights(self, insights: List[CallInsight]):
        """
        Add new insights to our historical dataset.
        The more data we have, the better our predictions become.
        """
        self.historical_insights.extend(insights)
        
        # Retrain models if we have enough data
        if len(self.historical_insights) >= 50:  # Minimum for meaningful predictions
            self._train_models()
    
    def _train_models(self):
        """
        Train multiple prediction models on our historical data.
        We're building several specialized forecasters for different aspects
        of the business - call volume, service demand, staffing needs, etc.
        """
        
        # Convert insights to feature matrix
        df = self._insights_to_dataframe(self.historical_insights)
        
        if len(df) < 10:  # Not enough data yet
            return
        
        # Feature engineering for time-based predictions
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Aggregate daily metrics
        daily_metrics = df.groupby(df['timestamp'].dt.date).agg({
            'call_id': 'count',  # calls per day
            'urgency_level': 'mean',  # average urgency
            'sentiment_score': 'mean',  # average satisfaction
            'revenue_opportunity': 'sum'  # total revenue potential
        }).rename(columns={'call_id': 'call_volume'})
        
        # Prepare features for prediction
        features = ['hour', 'day_of_week', 'month', 'is_weekend']
        feature_columns = []
        
        # Rolling averages for trend analysis
        for window in [7, 14, 30]:  # 1 week, 2 weeks, 1 month
            if len(daily_metrics) > window:
                daily_metrics[f'call_volume_avg_{window}d'] = daily_metrics['call_volume'].rolling(window).mean()
                daily_metrics[f'sentiment_avg_{window}d'] = daily_metrics['sentiment_score'].rolling(window).mean()
                feature_columns.extend([f'call_volume_avg_{window}d', f'sentiment_avg_{window}d'])
        
        # Train call volume predictor
        if len(daily_metrics) > 20:
            try:
                X = daily_metrics[feature_columns].fillna(method='forward').fillna(method='backward')
                y = daily_metrics['call_volume']
                
                # Remove rows with NaN values
                mask = ~(X.isna().any(axis=1) | y.isna())
                X_clean = X[mask]
                y_clean = y[mask]
                
                if len(X_clean) > 10:
                    # Scale features
                    self.scalers['call_volume'] = StandardScaler()
                    X_scaled = self.scalers['call_volume'].fit_transform(X_clean)
                    
                    # Train random forest for call volume prediction
                    self.models['call_volume'] = RandomForestRegressor(
                        n_estimators=50, 
                        random_state=42,
                        max_depth=10
                    )
                    self.models['call_volume'].fit(X_scaled, y_clean)
                    
                    # Train similar models for other metrics
                    for metric in ['sentiment_score', 'revenue_opportunity']:
                        if metric in daily_metrics.columns:
                            y_metric = daily_metrics[metric][mask]
                            self.models[metric] = RandomForestRegressor(
                                n_estimators=50, 
                                random_state=42,
                                max_depth=10
                            )
                            self.models[metric].fit(X_scaled, y_metric)
                    
                    self.is_trained = True
                    logging.info(f"Trained prediction models on {len(X_clean)} data points")
                    
            except Exception as e:
                logging.error(f"Error training models: {e}")
    
    def predict_next_week(self) -> Dict[str, Any]:
        """
        Generate predictions for the next 7 days.
        This is where the magic happens - turning historical patterns
        into actionable business intelligence.
        """
        
        if not self.is_trained:
            return {"error": "Models not trained yet. Need more historical data."}
        
        predictions = {
            "daily_forecasts": [],
            "weekly_totals": {},
            "recommendations": [],
            "confidence_scores": {}
        }
        
        # Generate daily predictions for next week
        for i in range(7):
            future_date = datetime.now().date() + timedelta(days=i+1)
            day_features = self._extract_date_features(future_date)
            
            daily_pred = {}
            for metric, model in self.models.items():
                if metric in self.scalers:
                    try:
                        # Create feature vector for this date
                        feature_vector = np.array([day_features]).reshape(1, -1)
                        
                        # Make prediction
                        if metric == 'call_volume':
                            scaled_features = self.scalers[metric].transform(feature_vector)
                            prediction = model.predict(scaled_features)[0]
                            daily_pred[metric] = max(0, int(round(prediction)))
                        else:
                            scaled_features = self.scalers['call_volume'].transform(feature_vector)
                            prediction = model.predict(scaled_features)[0]
                            daily_pred[metric] = prediction
                    except Exception as e:
                        logging.error(f"Prediction error for {metric}: {e}")
                        daily_pred[metric] = 0
            
            daily_pred['date'] = future_date.isoformat()
            predictions["daily_forecasts"].append(daily_pred)
        
        # Calculate weekly totals
        predictions["weekly_totals"] = {
            "total_calls": sum(day.get("call_volume", 0) for day in predictions["daily_forecasts"]),
            "avg_sentiment": np.mean([day.get("sentiment_score", 0) for day in predictions["daily_forecasts"]]),
            "total_revenue_opportunity": sum(day.get("revenue_opportunity", 0) for day in predictions["daily_forecasts"])
        }
        
        # Generate actionable recommendations
        predictions["recommendations"] = self._generate_recommendations(predictions)
        
        return predictions
    
    def _generate_recommendations(self, predictions: Dict) -> List[str]:
        """
        Convert predictions into specific, actionable business recommendations.
        No more "interesting insights" - we want "do this tomorrow" advice.
        """
        
        recommendations = []
        
        # Call volume recommendations
        weekly_calls = predictions["weekly_totals"]["total_calls"]
        if weekly_calls > 0:
            daily_avg = weekly_calls / 7
            
            # Check for high-volume days
            high_volume_days = [
                day for day in predictions["daily_forecasts"] 
                if day.get("call_volume", 0) > daily_avg * 1.3
            ]
            
            if high_volume_days:
                dates = [day["date"] for day in high_volume_days]
                recommendations.append(
                    f"HIGH CALL VOLUME ALERT: Schedule extra reception staff on {', '.join(dates)}. "
                    f"Expected {max(day.get('call_volume', 0) for day in high_volume_days)} calls on peak day "
                    f"(+{((max(day.get('call_volume', 0) for day in high_volume_days) / daily_avg - 1) * 100):.0f}% above average)."
                )
            
            # Low volume opportunities
            low_volume_days = [
                day for day in predictions["daily_forecasts"] 
                if day.get("call_volume", 0) < daily_avg * 0.7
            ]
            
            if low_volume_days:
                dates = [day["date"] for day in low_volume_days]
                recommendations.append(
                    f"MARKETING OPPORTUNITY: Lower call volume predicted on {', '.join(dates)}. "
                    f"Consider launching promotional campaigns to drive appointment bookings."
                )
        
        # Sentiment recommendations
        avg_sentiment = predictions["weekly_totals"]["avg_sentiment"]
        if avg_sentiment < -0.2:
            recommendations.append(
                f"QUALITY ALERT: Below-average customer satisfaction predicted (score: {avg_sentiment:.2f}). "
                f"Review staff training materials and consider customer service refresher sessions."
            )
        
        # Revenue opportunity recommendations
        total_revenue_opp = predictions["weekly_totals"]["total_revenue_opportunity"]
        if total_revenue_opp > 10000:
            recommendations.append(
                f"REVENUE OPPORTUNITY: ${total_revenue_opp:,.0f} in potential revenue identified. "
                f"Focus on conversion training and follow-up procedures to maximize bookings."
            )
        
        return recommendations
    
    def _extract_date_features(self, date) -> List[float]:
        """Extract features from a date for prediction."""
        dt = datetime.combine(date, datetime.min.time())
        return [
            dt.hour if hasattr(dt, 'hour') else 12,  # Default to noon
            dt.weekday(),
            dt.month,
            1.0 if dt.weekday() >= 5 else 0.0  # is_weekend
        ]
    
    def _insights_to_dataframe(self, insights: List[CallInsight]) -> pd.DataFrame:
        """Convert insights to pandas DataFrame for analysis."""
        data = []
        for insight in insights:
            data.append({
                'call_id': insight.call_id,
                'timestamp': insight.timestamp,
                'clinic_mentioned': insight.clinic_mentioned,
                'primary_intent': insight.primary_intent,
                'sentiment_score': insight.sentiment_score,
                'urgency_level': insight.urgency_level,
                'revenue_opportunity': insight.revenue_opportunity,
                'confidence_score': insight.confidence_score
            })
        return pd.DataFrame(data)
    
    def _load_historical_data(self, path: str) -> List[CallInsight]:
        """Load historical insights from file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            insights = []
            for item in data:
                insight = CallInsight(**item)
                insights.append(insight)
            
            return insights
        except Exception as e:
            logging.error(f"Error loading historical data: {e}")
            return []

class RealTimeCoachingSystem:
    """
    Phase 3: Real-time coaching features
    
    Borrowing from sports analytics here - coaches get real-time stats
    during games to make better decisions. Same principle for call centers.
    This could transform staff training from periodic reviews to continuous improvement.
    """
    
    def __init__(self, coaching_rules_path: Optional[str] = None):
        self.coaching_rules = self._load_coaching_rules(coaching_rules_path)
        self.active_sessions = {}  # Track ongoing calls
        self.performance_baselines = {}  # Staff performance benchmarks
        
        # Quality thresholds for alerts
        self.quality_thresholds = {
            "sentiment_drop": -0.3,  # Alert if sentiment drops this much
            "urgency_spike": 4,      # Alert if urgency reaches this level
            "call_duration": 900,    # Alert if call exceeds 15 minutes
            "resolution_rate": 0.8   # Expected resolution rate
        }
    
    async def start_coaching_session(self, staff_id: str, call_id: str) -> Dict:
        """
        Initialize real-time coaching for a new call.
        Sets up monitoring and establishes baseline expectations.
        """
        
        session = {
            "staff_id": staff_id,
            "call_id": call_id,
            "start_time": datetime.now(),
            "transcript_buffer": "",
            "alerts_triggered": [],
            "coaching_suggestions": [],
            "performance_metrics": {
                "empathy_score": 0.0,
                "efficiency_score": 0.0,
                "resolution_progress": 0.0
            }
        }
        
        self.active_sessions[call_id] = session
        
        # Get staff baseline performance for comparison
        baseline = self.performance_baselines.get(staff_id, {
            "avg_sentiment": 0.5,
            "avg_resolution_rate": 0.8,
            "avg_call_duration": 300
        })
        
        return {
            "session_id": call_id,
            "baseline_performance": baseline,
            "initial_coaching_tips": self._get_pre_call_tips(staff_id),
            "quality_targets": self.quality_thresholds
        }
    
    async def process_live_transcript_chunk(self, call_id: str, transcript_chunk: str) -> Dict:
        """
        Analyze incoming transcript chunks in real-time and provide
        immediate coaching feedback. This is like having a supervisor
        who can listen to every call simultaneously and provide instant,
        personalized coaching without being intrusive.
        """
        
        if call_id not in self.active_sessions:
            return {"error": "No active coaching session found"}
        
        session = self.active_sessions[call_id]
        session["transcript_buffer"] += " " + transcript_chunk
        
        # Analyze current conversation state
        analysis = await self._analyze_conversation_state(session["transcript_buffer"])
        
        # Generate real-time coaching
        coaching_response = {
            "timestamp": datetime.now().isoformat(),
            "immediate_alerts": [],
            "coaching_suggestions": [],
            "performance_updates": {},
            "script_recommendations": [],
            "escalation_flags": []
        }
        
        # Check for immediate alerts
        alerts = self._check_immediate_alerts(analysis, session)
        coaching_response["immediate_alerts"] = alerts
        
        # Generate coaching suggestions
        suggestions = self._generate_coaching_suggestions(analysis, session)
        coaching_response["coaching_suggestions"] = suggestions
        
        # Recommend script responses
        scripts = self._suggest_script_responses(analysis, session)
        coaching_response["script_recommendations"] = scripts
        
        # Check for escalation needs
        escalation_flags = self._check_escalation_triggers(analysis, session)
        coaching_response["escalation_flags"] = escalation_flags
        
        # Update session state
        session["alerts_triggered"].extend(alerts)
        session["coaching_suggestions"].extend(suggestions)
        
        return coaching_response
    
    async def _analyze_conversation_state(self, transcript: str) -> Dict:
        """
        Quick analysis of current conversation state for real-time coaching.
        Focused on immediate actionable insights rather than comprehensive analysis.
        """
        
        # Real-time analysis using lightweight heuristics for speed
        # In production, this could use a faster AI model or cached patterns
        
        analysis = {
            "current_sentiment": self._quick_sentiment_analysis(transcript),
            "detected_concerns": self._detect_immediate_concerns(transcript),
            "conversation_stage": self._identify_conversation_stage(transcript),
            "staff_performance_indicators": self._assess_staff_response(transcript),
            "customer_satisfaction_indicators": self._assess_customer_state(transcript)
        }
        
        return analysis
    
    def _check_immediate_alerts(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Check for conditions that require immediate attention.
        These are red-flag situations that need instant intervention.
        """
        
        alerts = []
        
        # Sentiment deterioration
        if analysis["current_sentiment"] < self.quality_thresholds["sentiment_drop"]:
            alerts.append(
                f"🚨 SENTIMENT ALERT: Customer satisfaction dropping rapidly "
                f"(current: {analysis['current_sentiment']:.2f}). Consider empathy response."
            )
        
        # Customer expressing frustration
        frustration_keywords = ["frustrated", "angry", "upset", "terrible", "awful"]
        if any(keyword in session["transcript_buffer"].lower() for keyword in frustration_keywords):
            alerts.append(
                "⚠️ FRUSTRATION DETECTED: Customer expressing negative emotions. "
                "Recommend acknowledgment and de-escalation techniques."
            )
        
        # Long call duration
        call_duration = (datetime.now() - session["start_time"]).total_seconds()
        if call_duration > self.quality_thresholds["call_duration"]:
            alerts.append(
                f"⏰ DURATION ALERT: Call exceeding {self.quality_thresholds['call_duration']/60:.0f} minutes. "
                f"Consider summarizing next steps or scheduling follow-up."
            )
        
        # Emergency keywords
        emergency_keywords = ["emergency", "severe pain", "bleeding", "accident", "urgent"]
        if any(keyword in session["transcript_buffer"].lower() for keyword in emergency_keywords):
            alerts.append(
                "🚨 EMERGENCY INDICATORS: Urgent medical situation detected. "
                "Prioritize immediate scheduling or emergency protocol."
            )
        
        return alerts
    
    def _generate_coaching_suggestions(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Generate specific, actionable coaching suggestions based on
        current conversation analysis. These are improvement opportunities
        that can be applied immediately or in future calls.
        """
        
        suggestions = []
        
        # Empathy coaching
        if analysis["current_sentiment"] < 0 and "empathy" not in session["coaching_suggestions"]:
            suggestions.append(
                "💝 EMPATHY OPPORTUNITY: Customer seems concerned. Try: "
                "'I understand your concern about [specific issue]. Let me help you with that.'"
            )
        
        # Information gathering
        if "question" in analysis["conversation_stage"]:
            suggestions.append(
                "❓ ACTIVE LISTENING: Customer asking questions. Make sure to: "
                "1) Fully answer their question, 2) Ask if they need clarification, "
                "3) Identify underlying concerns."
            )
        
        # Upselling opportunity
        if "cleaning" in session["transcript_buffer"].lower() and analysis["current_sentiment"] > 0.3:
            suggestions.append(
                "💰 UPSELL OPPORTUNITY: Customer booking cleaning. Consider mentioning: "
                "'While you're here, we could also check for any areas that might need attention.'"
            )
        
        # Closing techniques
        if analysis["conversation_stage"] == "resolution":
            suggestions.append(
                "🎯 CLOSING OPPORTUNITY: Ready to close the call. Ensure: "
                "1) All questions answered, 2) Next steps clear, 3) Customer satisfied."
            )
        
        return suggestions
    
    def _suggest_script_responses(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Suggest specific phrases or responses based on conversation context.
        These are ready-to-use responses that staff can adapt immediately.
        """
        
        scripts = []
        
        # Based on detected concerns
        concerns = analysis.get("detected_concerns", [])
        
        if "pricing" in concerns:
            scripts.append(
                "PRICING SCRIPT: 'I understand cost is important. We offer flexible payment "
                "plans and our treatment coordinator can discuss options that work for your budget.'"
            )
        
        if "availability" in concerns:
            scripts.append(
                "AVAILABILITY SCRIPT: 'Let me check our schedule for the next few weeks. "
                "We also maintain a cancellation list if you'd like earlier availability.'"
            )
        
        if "insurance" in concerns:
            scripts.append(
                "INSURANCE SCRIPT: 'We work with most major insurance plans and handle "
                "direct billing. I can verify your coverage and provide an estimate before your visit.'"
            )
        
        # Sentiment-based responses
        if analysis["current_sentiment"] < -0.2:
            scripts.append(
                "DE-ESCALATION SCRIPT: 'I apologize for any frustration. Your experience "
                "is important to us. Let me personally ensure we resolve this for you today.'"
            )
        
        return scripts
    
    def _check_escalation_triggers(self, analysis: Dict, session: Dict) -> List[str]:
        """
        Identify situations that may require supervisor intervention.
        Early warning system for complex or problematic calls.
        """
        
        escalation_flags = []
        
        # Multiple failed resolution attempts
        resolution_attempts = session["transcript_buffer"].lower().count("let me")
        if resolution_attempts > 3:
            escalation_flags.append(
                f"ESCALATION SUGGESTED: Multiple resolution attempts detected ({resolution_attempts}). "
                f"Consider supervisor involvement for complex issue resolution."
            )
        
        # Repeated customer frustration
        frustration_count = sum(1 for word in ["frustrated", "angry", "upset"] 
                             if word in session["transcript_buffer"].lower())
        if frustration_count > 2:
            escalation_flags.append(
                "ESCALATION REQUIRED: Repeated customer frustration expressions. "
                "Immediate supervisor intervention recommended."
            )
        
        # Legal or compliance keywords
        legal_keywords = ["lawyer", "legal", "sue", "complaint", "report"]
        if any(keyword in session["transcript_buffer"].lower() for keyword in legal_keywords):
            escalation_flags.append(
                "LEGAL ESCALATION: Legal language detected. "
                "Transfer to management immediately and document conversation."
            )
        
        return escalation_flags
    
    def _load_coaching_rules(self, path: Optional[str]) -> Dict:
        """Load coaching rules from configuration file."""
        if not path or not os.path.exists(path):
            return self._get_default_coaching_rules()
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading coaching rules: {e}")
            return self._get_default_coaching_rules()
    
    def _get_default_coaching_rules(self) -> Dict:
        """Default coaching rules for the system."""
        return {
            "sentiment_thresholds": {
                "excellent": 0.7,
                "good": 0.3,
                "neutral": 0.0,
                "concerning": -0.3,
                "critical": -0.7
            },
            "response_scripts": {
                "empathy": [
                    "I understand how you feel about this situation.",
                    "That must be frustrating for you.",
                    "I can see why that would be concerning."
                ],
                "reassurance": [
                    "We'll take care of this for you right away.",
                    "You're in good hands with our team.",
                    "We have extensive experience with this type of situation."
                ]
            }
        }
    
    def _get_pre_call_tips(self, staff_id: str) -> List[str]:
        """Get personalized pre-call coaching tips for staff member."""
        # This would be personalized based on staff performance history
        return [
            "Remember to use the customer's name during the conversation",
            "Ask open-ended questions to understand their full situation",
            "Confirm appointment details at the end of the call"
        ]
    
    def _quick_sentiment_analysis(self, transcript: str) -> float:
        """Quick sentiment scoring for real-time analysis."""
        positive_words = ["good", "great", "thank", "helpful", "excellent", "satisfied"]
        negative_words = ["bad", "terrible", "awful", "frustrated", "angry", "upset"]
        
        pos_count = sum(1 for word in positive_words if word in transcript.lower())
        neg_count = sum(1 for word in negative_words if word in transcript.lower())
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)
    
    def _detect_immediate_concerns(self, transcript: str) -> List[str]:
        """Detect immediate customer concerns for coaching."""
        concerns = []
        
        if any(word in transcript.lower() for word in ["price", "cost", "expensive", "afford"]):
            concerns.append("pricing")
        
        if any(word in transcript.lower() for word in ["available", "schedule", "appointment", "booking"]):
            concerns.append("availability")
        
        if any(word in transcript.lower() for word in ["insurance", "coverage", "benefits"]):
            concerns.append("insurance")
        
        if any(word in transcript.lower() for word in ["pain", "hurt", "emergency", "urgent"]):
            concerns.append("medical_urgency")
        
        return concerns
    
    def _identify_conversation_stage(self, transcript: str) -> str:
        """Identify current stage of conversation."""
        if "?" in transcript[-100:]:  # Recent questions
            return "question"
        elif any(word in transcript.lower() for word in ["thank", "bye", "goodbye"]):
            return "resolution"
        elif len(transcript) < 100:
            return "opening"
        else:
            return "discussion"
    
    def _assess_staff_response(self, transcript: str) -> Dict[str, float]:
        """Quick assessment of staff performance indicators."""
        return {
            "empathy_shown": 0.8 if any(word in transcript.lower() for word in ["understand", "sorry", "apologize"]) else 0.3,
            "information_provided": 0.7 if len(transcript.split()) > 50 else 0.4,
            "professional_tone": 0.9 if not any(word in transcript.lower() for word in ["um", "uh", "like"]) else 0.6
        }
    
    def _assess_customer_state(self, transcript: str) -> Dict[str, float]:
        """Quick assessment of customer satisfaction indicators."""
        sentiment = self._quick_sentiment_analysis(transcript)
        
        return {
            "satisfaction_level": sentiment,
            "engagement_level": 0.8 if "?" in transcript else 0.5,
            "resolution_likelihood": 0.9 if sentiment > 0.3 else 0.4
        }