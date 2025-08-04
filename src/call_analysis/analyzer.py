"""
Semantic transcript analyzer for the call analysis system.

This module provides AI-powered semantic analysis of call transcripts using OpenAI's
language models to extract structured insights about customer interactions.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import tiktoken
from openai import AsyncOpenAI

from .config import get_settings
from .models import CallInsight

logger = logging.getLogger(__name__)


class SemanticTranscriptAnalyzer:
    """
    AI-powered semantic analysis of call transcripts.
    
    This class replaces simple regex matching with sophisticated language model
    analysis to understand the true meaning and context of customer conversations.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the semantic analyzer.
        
        Args:
            api_key: OpenAI API key. If None, will use settings.
            model: Model to use. If None, will use settings.
        """
        settings = get_settings()
        
        # Initialize OpenAI client
        self.api_key = api_key or settings.openai.api_key
        self.model = model or settings.openai.model
        self.temperature = settings.openai.temperature
        self.max_tokens = settings.openai.max_tokens
        
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
        
        # Cache for analysis results to avoid redundant API calls
        self.analysis_cache: Dict[str, CallInsight] = {}
        
        # Define the analysis prompt template
        self.analysis_prompt_template = """
        You are an expert dental practice analyst with years of experience analyzing customer service calls.
        Analyze this call transcript and extract detailed, structured insights that will help improve operations.
        
        Call Transcript:
        {transcript}
        
        Extract the following information as valid JSON. Be precise and evidence-based in your analysis:
        
        {{
            "primary_intent": "appointment_booking|emergency|inquiry|complaint|cancellation|insurance|pricing|services|follow_up|referral",
            "secondary_intents": ["list", "of", "additional", "intents"],
            "sentiment_score": -1.0 to 1.0,
            "emotional_journey": {{
                "beginning": -1.0 to 1.0,
                "middle": -1.0 to 1.0, 
                "end": -1.0 to 1.0
            }},
            "urgency_level": 1 to 5,
            "resolution_status": "resolved|escalated|follow_up_needed|unresolved|pending",
            "staff_performance": {{
                "empathy": 0.0 to 1.0,
                "efficiency": 0.0 to 1.0,
                "knowledge": 0.0 to 1.0,
                "professionalism": 0.0 to 1.0
            }},
            "revenue_opportunity": estimated_dollar_value,
            "service_requests": ["cleaning", "filling", "crown", "extraction", "consultation", "etc"],
            "pain_points": ["wait_time", "pricing", "availability", "insurance", "pain", "etc"],
            "success_factors": ["friendly_staff", "quick_response", "knowledgeable", "flexible", "etc"],
            "next_actions": ["schedule_appointment", "send_pricing", "follow_up_call", "insurance_verification", "etc"],
            "confidence_score": 0.0 to 1.0,
            "clinic_mentioned": "clinic_name_if_mentioned_or_unknown"
        }}
        
        Guidelines:
        - Base sentiment on the overall tone and customer satisfaction indicators
        - Emotional journey should track how the customer's mood changed throughout the call
        - Urgency level: 1=routine, 2=preferred soon, 3=within a week, 4=urgent, 5=emergency
        - Revenue opportunity should be realistic based on services discussed
        - Staff performance should reflect actual demonstration of these qualities
        - Be conservative with confidence scores - only use high scores when very certain
        """
    
    async def analyze_transcript(self, transcript: str, call_id: str) -> CallInsight:
        """
        Analyze a call transcript and extract structured insights.
        
        Args:
            transcript: The call transcript text
            call_id: Unique identifier for the call
            
        Returns:
            CallInsight object with extracted information
            
        Raises:
            Exception: If analysis fails and fallback is not possible
        """
        # Check cache first to avoid redundant API calls
        cache_key = f"{call_id}_{hash(transcript[:200])}"  # Use first 200 chars for cache key
        if cache_key in self.analysis_cache:
            logger.debug(f"Using cached analysis for call {call_id}")
            return self.analysis_cache[cache_key]
        
        try:
            # Prepare transcript for analysis
            processed_transcript = self._prepare_transcript(transcript)
            
            # Create the analysis prompt
            prompt = self.analysis_prompt_template.format(transcript=processed_transcript)
            
            # Make API call to OpenAI
            logger.debug(f"Analyzing transcript for call {call_id}")
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert dental practice analyst. Always respond with valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=60.0,  # 60 second timeout
            )
            
            # Parse the response
            analysis_json = self._parse_ai_response(response.choices[0].message.content)
            
            # Create structured insight object
            insight = self._create_insight_from_analysis(call_id, analysis_json)
            
            # Cache the result
            self.analysis_cache[cache_key] = insight
            
            logger.info(f"Successfully analyzed call {call_id} with confidence {insight.confidence_score:.2f}")
            return insight
            
        except Exception as e:
            logger.error(f"Error analyzing transcript {call_id}: {e}")
            
            # Return fallback insight instead of raising
            return self._create_fallback_insight(call_id, transcript)
    
    def _prepare_transcript(self, transcript: str) -> str:
        """
        Prepare transcript for analysis by handling length limits and formatting.
        
        Args:
            transcript: Raw transcript text
            
        Returns:
            Processed transcript suitable for AI analysis
        """
        # Clean up the transcript
        transcript = transcript.strip()
        
        # Check token count and chunk if necessary
        max_tokens = 3000  # Leave room for prompt and response
        transcript_tokens = len(self.encoding.encode(transcript))
        
        if transcript_tokens > max_tokens:
            logger.warning(f"Transcript too long ({transcript_tokens} tokens), using intelligent chunking")
            transcript = self._intelligent_chunking(transcript, max_tokens)
        
        return transcript
    
    def _intelligent_chunking(self, transcript: str, max_tokens: int) -> str:
        """
        Intelligently chunk transcript to preserve conversation flow.
        
        Takes the beginning and end of the conversation to capture context
        and resolution, which are most important for analysis.
        
        Args:
            transcript: Original transcript
            max_tokens: Maximum tokens allowed
            
        Returns:
            Chunked transcript
        """
        tokens = self.encoding.encode(transcript)
        
        if len(tokens) <= max_tokens:
            return transcript
        
        # Take first 40% and last 40% of the conversation
        # This captures initial context and final resolution
        chunk_size = max_tokens // 2
        first_chunk = self.encoding.decode(tokens[:chunk_size])
        last_chunk = self.encoding.decode(tokens[-chunk_size:])
        
        return f"{first_chunk}\n\n[... middle of conversation omitted for length ...]\n\n{last_chunk}"
    
    def _parse_ai_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse AI response into structured data.
        
        Args:
            response_content: Raw response from AI
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Try to parse as JSON
            return json.loads(response_content)
        except json.JSONDecodeError as e:
            # Try to extract JSON from markdown code blocks
            if "```json" in response_content:
                json_start = response_content.find("```json") + 7
                json_end = response_content.find("```", json_start)
                json_content = response_content[json_start:json_end].strip()
                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Failed to parse AI response: {e}")
            logger.debug(f"Response content: {response_content}")
            raise ValueError(f"Could not parse AI response as JSON: {e}")
    
    def _create_insight_from_analysis(self, call_id: str, analysis: Dict[str, Any]) -> CallInsight:
        """
        Create CallInsight object from parsed analysis.
        
        Args:
            call_id: Unique call identifier
            analysis: Parsed analysis data
            
        Returns:
            CallInsight object
        """
        return CallInsight(
            call_id=call_id,
            timestamp=datetime.now(),
            clinic_mentioned=analysis.get("clinic_mentioned", "unknown"),
            primary_intent=analysis.get("primary_intent", "inquiry"),
            secondary_intents=analysis.get("secondary_intents", []),
            sentiment_score=float(analysis.get("sentiment_score", 0.0)),
            emotional_journey=analysis.get("emotional_journey", {}),
            urgency_level=int(analysis.get("urgency_level", 1)),
            resolution_status=analysis.get("resolution_status", "unknown"),
            staff_performance=analysis.get("staff_performance", {}),
            revenue_opportunity=float(analysis.get("revenue_opportunity", 0.0)),
            service_requests=analysis.get("service_requests", []),
            pain_points=analysis.get("pain_points", []),
            success_factors=analysis.get("success_factors", []),
            next_actions=analysis.get("next_actions", []),
            confidence_score=float(analysis.get("confidence_score", 0.0))
        )
    
    def _create_fallback_insight(self, call_id: str, transcript: str) -> CallInsight:
        """
        Create fallback insight when AI analysis fails.
        
        Uses simple heuristics to provide basic analysis rather than failing completely.
        
        Args:
            call_id: Unique call identifier
            transcript: Original transcript
            
        Returns:
            Basic CallInsight object
        """
        logger.warning(f"Using fallback analysis for call {call_id}")
        
        # Basic sentiment analysis using keyword counting
        positive_words = ["good", "great", "thank", "helpful", "satisfied", "excellent", "wonderful"]
        negative_words = ["bad", "terrible", "awful", "frustrated", "angry", "upset", "disappointed"]
        
        transcript_lower = transcript.lower()
        pos_count = sum(1 for word in positive_words if word in transcript_lower)
        neg_count = sum(1 for word in negative_words if word in transcript_lower)
        
        # Calculate basic sentiment
        total_sentiment_words = pos_count + neg_count
        if total_sentiment_words > 0:
            sentiment = (pos_count - neg_count) / total_sentiment_words
        else:
            sentiment = 0.0
        
        # Detect basic intent
        intent_keywords = {
            "appointment_booking": ["appointment", "schedule", "book", "available"],
            "emergency": ["emergency", "urgent", "pain", "bleeding"],
            "pricing": ["cost", "price", "insurance", "payment"],
            "complaint": ["complaint", "problem", "issue", "dissatisfied"],
        }
        
        primary_intent = "inquiry"  # default
        for intent, keywords in intent_keywords.items():
            if any(keyword in transcript_lower for keyword in keywords):
                primary_intent = intent
                break
        
        # Basic urgency detection
        urgency = 1
        if any(word in transcript_lower for word in ["urgent", "emergency", "asap", "immediately"]):
            urgency = 5
        elif any(word in transcript_lower for word in ["soon", "quickly", "prefer"]):
            urgency = 3
        
        return CallInsight(
            call_id=call_id,
            timestamp=datetime.now(),
            clinic_mentioned="unknown",
            primary_intent=primary_intent,
            secondary_intents=[],
            sentiment_score=sentiment,
            emotional_journey={
                "beginning": 0.0,
                "middle": sentiment,
                "end": sentiment
            },
            urgency_level=urgency,
            resolution_status="unknown",
            staff_performance={
                "empathy": 0.5,
                "efficiency": 0.5,
                "knowledge": 0.5,
                "professionalism": 0.5
            },
            revenue_opportunity=100.0,  # Conservative estimate
            service_requests=[],
            pain_points=["analysis_failed"],
            success_factors=[],
            next_actions=["manual_review_needed"],
            confidence_score=0.2  # Low confidence for fallback
        )
    
    async def analyze_batch(self, transcripts: List[Dict[str, str]]) -> List[CallInsight]:
        """
        Analyze multiple transcripts in batch.
        
        Args:
            transcripts: List of dicts with 'call_id' and 'transcript' keys
            
        Returns:
            List of CallInsight objects
        """
        insights = []
        
        for item in transcripts:
            try:
                insight = await self.analyze_transcript(
                    transcript=item["transcript"],
                    call_id=item["call_id"]
                )
                insights.append(insight)
            except Exception as e:
                logger.error(f"Failed to analyze call {item['call_id']}: {e}")
                # Add fallback insight for failed analysis
                fallback = self._create_fallback_insight(item["call_id"], item["transcript"])
                insights.append(fallback)
        
        return insights
    
    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the analysis cache."""
        return {
            "cache_size": len(self.analysis_cache),
            "cache_keys": list(self.analysis_cache.keys()),
        }
    
    async def validate_api_key(self) -> bool:
        """
        Validate the OpenAI API key by making a simple request.
        
        Returns:
            True if API key is valid, False otherwise
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use cheaper model for validation
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
                temperature=0
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False


# Factory function for easy instantiation
def create_analyzer(api_key: Optional[str] = None, model: Optional[str] = None) -> SemanticTranscriptAnalyzer:
    """
    Create a configured SemanticTranscriptAnalyzer instance.
    
    Args:
        api_key: OpenAI API key override
        model: Model name override
        
    Returns:
        Configured analyzer instance
    """
    return SemanticTranscriptAnalyzer(api_key=api_key, model=model)