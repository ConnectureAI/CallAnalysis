"""
Advanced NLP analyzer combining multiple sophisticated techniques.

This module orchestrates various NLP components to provide comprehensive
text analysis including entity extraction, topic modeling, sentiment analysis,
and linguistic feature extraction.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

import spacy
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import networkx as nx
from textstat import flesch_reading_ease, flesch_kincaid_grade

from ..config import get_settings
from .entity_extraction import EntityExtractor
from .topic_modeling import TopicModelingEngine
from .sentiment_analyzer import AdvancedSentimentAnalyzer
from .linguistic_features import LinguisticFeatureExtractor
from .text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class AdvancedNLPInsights:
    """Comprehensive NLP analysis results."""
    
    # Basic info
    call_id: str
    timestamp: datetime
    
    # Entity extraction
    named_entities: List[Dict[str, Any]]
    person_names: List[str]
    organizations: List[str]
    locations: List[str]
    dates: List[str]
    phone_numbers: List[str]
    medical_terms: List[str]
    
    # Topic modeling
    topics: List[Dict[str, Any]]
    topic_distribution: List[float]
    dominant_topic: str
    topic_coherence_score: float
    
    # Advanced sentiment
    sentiment_scores: Dict[str, float]
    emotion_analysis: Dict[str, float]
    sentiment_trajectory: List[Tuple[int, float]]  # (position, sentiment)
    
    # Linguistic features
    readability_scores: Dict[str, float]
    linguistic_complexity: Dict[str, float]
    discourse_markers: List[str]
    speech_patterns: Dict[str, Any]
    
    # Conversation analysis
    turn_taking_analysis: Dict[str, Any]
    interruption_patterns: List[Dict[str, Any]]
    conversation_flow: Dict[str, Any]
    
    # Business insights
    intent_probabilities: Dict[str, float]
    urgency_indicators: List[str]
    satisfaction_predictors: List[str]
    conversion_signals: List[str]
    
    # Semantic analysis
    semantic_similarity_scores: Dict[str, float]
    key_phrases: List[Dict[str, Any]]
    semantic_roles: List[Dict[str, Any]]
    
    # Quality metrics
    analysis_confidence: float
    processing_time: float


class AdvancedNLPAnalyzer:
    """
    Advanced NLP analyzer combining multiple sophisticated techniques.
    
    This class orchestrates various NLP components to provide comprehensive
    analysis of call transcripts using state-of-the-art techniques.
    """
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """
        Initialize the advanced NLP analyzer.
        
        Args:
            model_cache_dir: Directory to cache models
        """
        self.settings = get_settings()
        self.model_cache_dir = model_cache_dir or str(self.settings.models_dir / "nlp")
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.entity_extractor = EntityExtractor()
        self.topic_engine = TopicModelingEngine()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.feature_extractor = LinguisticFeatureExtractor()
        
        # Load models
        self._load_models()
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _load_models(self):
        """Load all required NLP models."""
        logger.info("Loading advanced NLP models...")
        
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Load transformer models
            self.bert_tokenizer = AutoTokenizer.from_pretrained(
                "bert-base-uncased", cache_dir=self.model_cache_dir
            )
            self.bert_model = AutoModel.from_pretrained(
                "bert-base-uncased", cache_dir=self.model_cache_dir
            )
            
            # Load specialized models
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                cache_dir=self.model_cache_dir
            )
            
            self.intent_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                cache_dir=self.model_cache_dir
            )
            
            logger.info("All NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            # Set up fallback modes
            self.nlp = None
            self.bert_model = None
            self.emotion_classifier = None
            self.intent_classifier = None
    
    async def analyze_transcript(
        self, 
        transcript: str, 
        call_id: str,
        speaker_labels: Optional[List[str]] = None
    ) -> AdvancedNLPInsights:
        """
        Perform comprehensive NLP analysis on transcript.
        
        Args:
            transcript: Call transcript text
            call_id: Unique call identifier
            speaker_labels: Optional speaker labels for each segment
            
        Returns:
            AdvancedNLPInsights object with comprehensive analysis
        """
        start_time = datetime.now()
        logger.info(f"Starting advanced NLP analysis for call {call_id}")
        
        try:
            # Preprocess text
            processed_text = await self._preprocess_text(transcript)
            
            # Run parallel analysis tasks
            analysis_tasks = [
                self._extract_entities(processed_text),
                self._analyze_topics(processed_text),
                self._analyze_sentiment_advanced(processed_text),
                self._extract_linguistic_features(processed_text),
                self._analyze_conversation_structure(transcript, speaker_labels),
                self._extract_business_insights(processed_text),
                self._perform_semantic_analysis(processed_text),
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Combine results
            insights = self._combine_results(
                call_id, start_time, processed_text, results
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            insights.processing_time = processing_time
            
            logger.info(f"Advanced NLP analysis completed for {call_id} in {processing_time:.2f}s")
            return insights
            
        except Exception as e:
            logger.error(f"Error in advanced NLP analysis for {call_id}: {e}")
            return self._create_fallback_insights(call_id, start_time)
    
    async def _preprocess_text(self, transcript: str) -> str:
        """Preprocess transcript text."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.preprocessor.preprocess, transcript
        )
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract named entities and medical terms."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.entity_extractor.extract_all, text
        )
    
    async def _analyze_topics(self, text: str) -> Dict[str, Any]:
        """Perform topic modeling analysis."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.topic_engine.analyze_topics, text
        )
    
    async def _analyze_sentiment_advanced(self, text: str) -> Dict[str, Any]:
        """Perform advanced sentiment and emotion analysis."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.sentiment_analyzer.analyze_comprehensive, text
        )
    
    async def _extract_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic and readability features."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self.feature_extractor.extract_all_features, text
        )
    
    async def _analyze_conversation_structure(
        self, transcript: str, speaker_labels: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Analyze conversation structure and turn-taking patterns."""
        def _analyze():
            return self._conversation_analysis(transcript, speaker_labels)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _analyze
        )
    
    async def _extract_business_insights(self, text: str) -> Dict[str, Any]:
        """Extract business-relevant insights."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._business_analysis, text
        )
    
    async def _perform_semantic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform semantic analysis using transformers."""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._semantic_analysis, text
        )
    
    def _conversation_analysis(
        self, transcript: str, speaker_labels: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Analyze conversation structure and patterns."""
        try:
            segments = transcript.split('\n')
            
            # Basic turn-taking analysis
            turn_count = 0
            avg_turn_length = 0
            interruptions = []
            
            if speaker_labels and len(speaker_labels) == len(segments):
                current_speaker = None
                turn_lengths = []
                
                for i, (segment, speaker) in enumerate(zip(segments, speaker_labels)):
                    if speaker != current_speaker:
                        turn_count += 1
                        current_speaker = speaker
                    
                    turn_lengths.append(len(segment.split()))
                    
                    # Detect potential interruptions (short segments)
                    if len(segment.split()) < 3 and i < len(segments) - 1:
                        interruptions.append({
                            "position": i,
                            "speaker": speaker,
                            "text": segment
                        })
                
                avg_turn_length = np.mean(turn_lengths) if turn_lengths else 0
            
            # Conversation flow analysis
            flow_metrics = self._analyze_conversation_flow(segments)
            
            return {
                "turn_taking_analysis": {
                    "total_turns": turn_count,
                    "average_turn_length": avg_turn_length,
                    "turn_distribution": self._calculate_turn_distribution(speaker_labels) if speaker_labels else {}
                },
                "interruption_patterns": interruptions,
                "conversation_flow": flow_metrics
            }
            
        except Exception as e:
            logger.error(f"Error in conversation analysis: {e}")
            return {
                "turn_taking_analysis": {},
                "interruption_patterns": [],
                "conversation_flow": {}
            }
    
    def _analyze_conversation_flow(self, segments: List[str]) -> Dict[str, Any]:
        """Analyze the flow and coherence of conversation."""
        if not segments:
            return {}
        
        # Calculate semantic similarity between adjacent segments
        similarities = []
        if self.bert_model and len(segments) > 1:
            for i in range(len(segments) - 1):
                sim = self._calculate_semantic_similarity(segments[i], segments[i + 1])
                similarities.append(sim)
        
        # Analyze topic shifts
        topic_shifts = self._detect_topic_shifts(segments)
        
        return {
            "coherence_score": np.mean(similarities) if similarities else 0.0,
            "topic_shifts": topic_shifts,
            "conversation_progression": self._analyze_progression(segments)
        }
    
    def _business_analysis(self, text: str) -> Dict[str, Any]:
        """Extract business-relevant insights."""
        try:
            # Intent classification
            intent_labels = [
                "appointment scheduling", "emergency", "inquiry", "complaint",
                "insurance question", "pricing", "service information", "cancellation"
            ]
            
            intent_probs = {}
            if self.intent_classifier:
                result = self.intent_classifier(text, intent_labels)
                for label, score in zip(result['labels'], result['scores']):
                    intent_probs[label] = score
            
            # Urgency detection
            urgency_indicators = self._detect_urgency_indicators(text)
            
            # Satisfaction predictors
            satisfaction_predictors = self._detect_satisfaction_predictors(text)
            
            # Conversion signals
            conversion_signals = self._detect_conversion_signals(text)
            
            return {
                "intent_probabilities": intent_probs,
                "urgency_indicators": urgency_indicators,
                "satisfaction_predictors": satisfaction_predictors,
                "conversion_signals": conversion_signals
            }
            
        except Exception as e:
            logger.error(f"Error in business analysis: {e}")
            return {
                "intent_probabilities": {},
                "urgency_indicators": [],
                "satisfaction_predictors": [],
                "conversion_signals": []
            }
    
    def _semantic_analysis(self, text: str) -> Dict[str, Any]:
        """Perform semantic analysis using BERT embeddings."""
        try:
            if not self.bert_model or not self.bert_tokenizer:
                return {"semantic_similarity_scores": {}, "key_phrases": [], "semantic_roles": []}
            
            # Generate embeddings
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Extract key phrases using attention weights
            key_phrases = self._extract_key_phrases_bert(text, embeddings)
            
            # Semantic role labeling (simplified)
            semantic_roles = self._extract_semantic_roles(text)
            
            # Calculate similarity with common business contexts
            business_contexts = [
                "scheduling appointment", "dental emergency", "insurance coverage",
                "treatment cost", "patient satisfaction", "service quality"
            ]
            
            similarity_scores = {}
            for context in business_contexts:
                sim_score = self._calculate_semantic_similarity(text, context)
                similarity_scores[context] = sim_score
            
            return {
                "semantic_similarity_scores": similarity_scores,
                "key_phrases": key_phrases,
                "semantic_roles": semantic_roles
            }
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return {"semantic_similarity_scores": {}, "key_phrases": [], "semantic_roles": []}
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.bert_model or not self.bert_tokenizer:
            return 0.0
        
        try:
            # Tokenize both texts
            inputs1 = self.bert_tokenizer(text1, return_tensors="pt", truncation=True, max_length=512)
            inputs2 = self.bert_tokenizer(text2, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                # Get embeddings
                outputs1 = self.bert_model(**inputs1)
                outputs2 = self.bert_model(**inputs2)
                
                # Use [CLS] token embeddings
                emb1 = outputs1.last_hidden_state[0, 0, :].numpy()
                emb2 = outputs2.last_hidden_state[0, 0, :].numpy()
                
                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                return float(similarity)
                
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {e}")
            return 0.0
    
    def _extract_key_phrases_bert(self, text: str, embeddings: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract key phrases using BERT attention."""
        try:
            tokens = self.bert_tokenizer.tokenize(text)
            
            # Simple approach: use embedding magnitude as importance score
            token_scores = torch.norm(embeddings[0], dim=1).numpy()
            
            # Get top scoring tokens
            top_indices = np.argsort(token_scores)[-10:]  # Top 10 tokens
            
            key_phrases = []
            for idx in top_indices:
                if idx < len(tokens):
                    key_phrases.append({
                        "phrase": tokens[idx],
                        "score": float(token_scores[idx]),
                        "position": int(idx)
                    })
            
            return sorted(key_phrases, key=lambda x: x["score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def _extract_semantic_roles(self, text: str) -> List[Dict[str, Any]]:
        """Extract semantic roles using spaCy."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text)
            roles = []
            
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ in ["nsubj", "dobj", "iobj", "agent"]:
                        roles.append({
                            "word": token.text,
                            "role": token.dep_,
                            "head": token.head.text,
                            "position": token.idx
                        })
            
            return roles
            
        except Exception as e:
            logger.error(f"Error extracting semantic roles: {e}")
            return []
    
    def _detect_urgency_indicators(self, text: str) -> List[str]:
        """Detect urgency indicators in text."""
        urgency_patterns = [
            "emergency", "urgent", "asap", "immediately", "right away",
            "severe pain", "bleeding", "can't wait", "need help now",
            "very important", "critical", "serious"
        ]
        
        found_indicators = []
        text_lower = text.lower()
        
        for pattern in urgency_patterns:
            if pattern in text_lower:
                found_indicators.append(pattern)
        
        return found_indicators
    
    def _detect_satisfaction_predictors(self, text: str) -> List[str]:
        """Detect satisfaction predictors."""
        positive_predictors = [
            "thank you", "appreciate", "helpful", "great service",
            "satisfied", "excellent", "professional", "friendly"
        ]
        
        negative_predictors = [
            "frustrated", "disappointed", "terrible", "awful",
            "unhappy", "dissatisfied", "poor service", "rude"
        ]
        
        found_predictors = []
        text_lower = text.lower()
        
        for predictor in positive_predictors + negative_predictors:
            if predictor in text_lower:
                found_predictors.append(predictor)
        
        return found_predictors
    
    def _detect_conversion_signals(self, text: str) -> List[str]:
        """Detect conversion signals."""
        conversion_signals = [
            "book appointment", "schedule", "when can I come in",
            "what times available", "sign me up", "interested in",
            "how much does it cost", "insurance cover", "payment plan"
        ]
        
        found_signals = []
        text_lower = text.lower()
        
        for signal in conversion_signals:
            if signal in text_lower:
                found_signals.append(signal)
        
        return found_signals
    
    def _calculate_turn_distribution(self, speaker_labels: List[str]) -> Dict[str, float]:
        """Calculate speaking time distribution."""
        if not speaker_labels:
            return {}
        
        from collections import Counter
        counts = Counter(speaker_labels)
        total = len(speaker_labels)
        
        return {speaker: count / total for speaker, count in counts.items()}
    
    def _detect_topic_shifts(self, segments: List[str]) -> List[Dict[str, Any]]:
        """Detect topic shifts in conversation."""
        topic_shifts = []
        
        # Simple approach: detect shifts based on vocabulary overlap
        for i in range(1, len(segments)):
            prev_words = set(segments[i-1].lower().split())
            curr_words = set(segments[i].lower().split())
            
            overlap = len(prev_words & curr_words)
            total_unique = len(prev_words | curr_words)
            
            if total_unique > 0:
                similarity = overlap / total_unique
                if similarity < 0.3:  # Low similarity indicates topic shift
                    topic_shifts.append({
                        "position": i,
                        "similarity": similarity,
                        "shift_type": "major" if similarity < 0.1 else "minor"
                    })
        
        return topic_shifts
    
    def _analyze_progression(self, segments: List[str]) -> Dict[str, Any]:
        """Analyze conversation progression."""
        if not segments:
            return {}
        
        # Analyze sentiment progression
        sentiment_progression = []
        for i, segment in enumerate(segments):
            # Simple sentiment scoring
            positive_words = ["good", "great", "thank", "yes", "sure", "perfect"]
            negative_words = ["no", "bad", "terrible", "problem", "issue", "wrong"]
            
            pos_count = sum(1 for word in positive_words if word in segment.lower())
            neg_count = sum(1 for word in negative_words if word in segment.lower())
            
            sentiment = (pos_count - neg_count) / max(pos_count + neg_count, 1)
            sentiment_progression.append(sentiment)
        
        return {
            "sentiment_trajectory": list(enumerate(sentiment_progression)),
            "overall_trend": "improving" if sentiment_progression[-1] > sentiment_progression[0] else "declining",
            "volatility": np.std(sentiment_progression) if len(sentiment_progression) > 1 else 0.0
        }
    
    def _combine_results(
        self, 
        call_id: str, 
        start_time: datetime, 
        text: str, 
        results: List[Any]
    ) -> AdvancedNLPInsights:
        """Combine all analysis results into insights object."""
        try:
            entities = results[0] if not isinstance(results[0], Exception) else {}
            topics = results[1] if not isinstance(results[1], Exception) else {}
            sentiment = results[2] if not isinstance(results[2], Exception) else {}
            linguistic = results[3] if not isinstance(results[3], Exception) else {}
            conversation = results[4] if not isinstance(results[4], Exception) else {}
            business = results[5] if not isinstance(results[5], Exception) else {}
            semantic = results[6] if not isinstance(results[6], Exception) else {}
            
            # Calculate overall confidence
            confidence_scores = []
            if entities:
                confidence_scores.append(0.9)
            if topics:
                confidence_scores.append(topics.get("topic_coherence_score", 0.5))
            if sentiment:
                confidence_scores.append(sentiment.get("confidence", 0.5))
            
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.3
            
            return AdvancedNLPInsights(
                call_id=call_id,
                timestamp=start_time,
                
                # Entities
                named_entities=entities.get("all_entities", []),
                person_names=entities.get("persons", []),
                organizations=entities.get("organizations", []),
                locations=entities.get("locations", []),
                dates=entities.get("dates", []),
                phone_numbers=entities.get("phone_numbers", []),
                medical_terms=entities.get("medical_terms", []),
                
                # Topics
                topics=topics.get("topics", []),
                topic_distribution=topics.get("topic_distribution", []),
                dominant_topic=topics.get("dominant_topic", "unknown"),
                topic_coherence_score=topics.get("topic_coherence_score", 0.0),
                
                # Sentiment
                sentiment_scores=sentiment.get("sentiment_scores", {}),
                emotion_analysis=sentiment.get("emotion_analysis", {}),
                sentiment_trajectory=sentiment.get("sentiment_trajectory", []),
                
                # Linguistic
                readability_scores=linguistic.get("readability", {}),
                linguistic_complexity=linguistic.get("complexity", {}),
                discourse_markers=linguistic.get("discourse_markers", []),
                speech_patterns=linguistic.get("speech_patterns", {}),
                
                # Conversation
                turn_taking_analysis=conversation.get("turn_taking_analysis", {}),
                interruption_patterns=conversation.get("interruption_patterns", []),
                conversation_flow=conversation.get("conversation_flow", {}),
                
                # Business
                intent_probabilities=business.get("intent_probabilities", {}),
                urgency_indicators=business.get("urgency_indicators", []),
                satisfaction_predictors=business.get("satisfaction_predictors", []),
                conversion_signals=business.get("conversion_signals", []),
                
                # Semantic
                semantic_similarity_scores=semantic.get("semantic_similarity_scores", {}),
                key_phrases=semantic.get("key_phrases", []),
                semantic_roles=semantic.get("semantic_roles", []),
                
                # Quality
                analysis_confidence=overall_confidence,
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error combining analysis results: {e}")
            return self._create_fallback_insights(call_id, start_time)
    
    def _create_fallback_insights(self, call_id: str, start_time: datetime) -> AdvancedNLPInsights:
        """Create fallback insights when analysis fails."""
        return AdvancedNLPInsights(
            call_id=call_id,
            timestamp=start_time,
            named_entities=[],
            person_names=[],
            organizations=[],
            locations=[],
            dates=[],
            phone_numbers=[],
            medical_terms=[],
            topics=[],
            topic_distribution=[],
            dominant_topic="unknown",
            topic_coherence_score=0.0,
            sentiment_scores={},
            emotion_analysis={},
            sentiment_trajectory=[],
            readability_scores={},
            linguistic_complexity={},
            discourse_markers=[],
            speech_patterns={},
            turn_taking_analysis={},
            interruption_patterns=[],
            conversation_flow={},
            intent_probabilities={},
            urgency_indicators=[],
            satisfaction_predictors=[],
            conversion_signals=[],
            semantic_similarity_scores={},
            key_phrases=[],
            semantic_roles=[],
            analysis_confidence=0.1,
            processing_time=0.0
        )
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)