"""
Intent Classification for call analysis.

This module provides sophisticated intent detection capabilities
specifically designed for dental practice customer interactions.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


class IntentClassifier:
    """
    Intent classification for dental practice calls.
    
    Classifies customer intents using rule-based patterns,
    keyword matching, and machine learning approaches.
    """
    
    def __init__(self):
        """Initialize intent classifier."""
        self._setup_nltk()
        self._load_intent_patterns()
        self._setup_ml_classifier()
    
    def _setup_nltk(self):
        """Setup NLTK components."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def _load_intent_patterns(self):
        """Load intent detection patterns."""
        # Dental practice specific intents with patterns
        self.intent_patterns = {
            "appointment_booking": {
                "keywords": [
                    "appointment", "schedule", "book", "reserve", "available",
                    "when can", "slot", "time", "date", "calendar", "booking"
                ],
                "phrases": [
                    "book an appointment", "schedule appointment", "make appointment",
                    "available times", "when can I come", "appointment slot",
                    "reserve time", "schedule visit"
                ],
                "patterns": [
                    r"(?:book|schedule|make).*appointment",
                    r"(?:when|what time).*(?:available|come|visit)",
                    r"(?:available|free).*(?:time|slot|appointment)",
                    r"(?:appointment|visit).*(?:today|tomorrow|next week)"
                ],
                "confidence_boost": 0.3
            },
            
            "appointment_cancellation": {
                "keywords": [
                    "cancel", "cancellation", "reschedule", "postpone", "change",
                    "move", "different time", "cannot make"
                ],
                "phrases": [
                    "cancel appointment", "cancel my appointment", "reschedule appointment",
                    "change appointment", "move appointment", "cannot make it"
                ],
                "patterns": [
                    r"(?:cancel|reschedule|change|move).*appointment",
                    r"(?:cannot|can't|unable).*(?:make|come|attend)",
                    r"(?:postpone|delay).*(?:appointment|visit)"
                ],
                "confidence_boost": 0.2
            },
            
            "emergency_dental": {
                "keywords": [
                    "emergency", "urgent", "pain", "severe", "immediate",
                    "asap", "right now", "help", "hurt", "ache", "bleeding"
                ],
                "phrases": [
                    "dental emergency", "urgent care", "severe pain", "tooth pain",
                    "need help", "hurts badly", "emergency appointment"
                ],
                "patterns": [
                    r"(?:emergency|urgent|immediate).*(?:dental|tooth|care)",
                    r"(?:severe|bad|terrible).*(?:pain|ache|hurt)",
                    r"(?:need|require).*(?:immediate|urgent|asap)",
                    r"(?:bleeding|swollen|infected).*(?:tooth|gum|mouth)"
                ],
                "confidence_boost": 0.4
            },
            
            "insurance_inquiry": {
                "keywords": [
                    "insurance", "coverage", "benefits", "claim", "copay",
                    "deductible", "in-network", "covered", "cost", "price"
                ],
                "phrases": [
                    "insurance coverage", "dental benefits", "how much cost",
                    "covered by insurance", "file claim", "copay amount"
                ],
                "patterns": [
                    r"(?:insurance|coverage|benefits).*(?:cover|pay|include)",
                    r"(?:how much|cost|price|expense).*(?:procedure|treatment)",
                    r"(?:copay|deductible|out of pocket).*(?:amount|cost)",
                    r"(?:in-network|covered|accepted).*(?:provider|insurance)"
                ],
                "confidence_boost": 0.25
            },
            
            "treatment_inquiry": {
                "keywords": [
                    "treatment", "procedure", "cleaning", "filling", "crown",
                    "root canal", "extraction", "whitening", "implant", "what is"
                ],
                "phrases": [
                    "what is procedure", "how does treatment work", "procedure information",
                    "treatment options", "what involves", "procedure steps"
                ],
                "patterns": [
                    r"(?:what is|explain|tell me about).*(?:procedure|treatment)",
                    r"(?:how does|how long).*(?:procedure|treatment|process)",
                    r"(?:treatment|procedure).*(?:options|available|possible)",
                    r"(?:need|require|recommend).*(?:treatment|procedure)"
                ],
                "confidence_boost": 0.2
            },
            
            "complaint": {
                "keywords": [
                    "complaint", "dissatisfied", "unhappy", "problem", "issue",
                    "disappointed", "poor service", "bad experience", "wrong"
                ],
                "phrases": [
                    "file complaint", "not satisfied", "poor service", "bad experience",
                    "something wrong", "not happy", "disappointed with"
                ],
                "patterns": [
                    r"(?:complaint|complain|issue|problem).*(?:service|treatment|experience)",
                    r"(?:not|un)(?:satisfied|happy).*(?:with|about)",
                    r"(?:poor|bad|terrible).*(?:service|experience|treatment)",
                    r"(?:disappointed|upset|frustrated).*(?:with|about)"
                ],
                "confidence_boost": 0.3
            },
            
            "billing_inquiry": {
                "keywords": [
                    "billing", "bill", "payment", "charge", "invoice",
                    "account", "balance", "owe", "paid", "receipt"
                ],
                "phrases": [
                    "billing question", "payment information", "account balance",
                    "invoice details", "charge explanation", "payment plan"
                ],
                "patterns": [
                    r"(?:billing|bill|invoice|charge).*(?:question|inquiry|problem)",
                    r"(?:payment|pay).*(?:plan|option|method|schedule)",
                    r"(?:account|balance).*(?:information|status|inquiry)",
                    r"(?:owe|charged|cost).*(?:how much|amount)"
                ],
                "confidence_boost": 0.2
            },
            
            "follow_up": {
                "keywords": [
                    "follow up", "check back", "after treatment", "post procedure",
                    "recovery", "healing", "progress", "how am I doing"
                ],
                "phrases": [
                    "follow up appointment", "check progress", "after procedure",
                    "recovery status", "healing well", "post treatment"
                ],
                "patterns": [
                    r"(?:follow up|check back|follow-up).*(?:appointment|visit|call)",
                    r"(?:after|post).*(?:treatment|procedure|surgery)",
                    r"(?:recovery|healing).*(?:progress|status|how)",
                    r"(?:check|see how).*(?:doing|healing|recovering)"
                ],
                "confidence_boost": 0.2
            },
            
            "general_inquiry": {
                "keywords": [
                    "information", "question", "ask", "wondering", "curious",
                    "tell me", "explain", "help", "service", "hours"
                ],
                "phrases": [
                    "general question", "information about", "office hours",
                    "services offered", "location information", "contact information"
                ],
                "patterns": [
                    r"(?:information|info|details).*(?:about|regarding)",
                    r"(?:office|clinic).*(?:hours|location|address)",
                    r"(?:services|treatments).*(?:offered|available|provide)",
                    r"(?:general|simple).*(?:question|inquiry)"
                ],
                "confidence_boost": 0.1
            }
        }
        
        # Urgency indicators
        self.urgency_indicators = {
            "high": [
                "emergency", "urgent", "asap", "immediately", "right now",
                "severe pain", "bleeding", "swollen", "infected", "can't eat"
            ],
            "medium": [
                "soon", "today", "this week", "pain", "uncomfortable",
                "bothering", "sensitive", "worried"
            ],
            "low": [
                "when convenient", "sometime", "eventually", "routine",
                "checkup", "cleaning", "non-urgent"
            ]
        }
    
    def _setup_ml_classifier(self):
        """Setup machine learning classifier (simplified)."""
        # In a real implementation, you would train this on labeled data
        self.ml_classifier = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def classify_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify the intent of the given text.
        
        Args:
            text: Input text to classify
            context: Additional context information
            
        Returns:
            Dictionary containing intent classification results
        """
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            
            # Rule-based classification
            rule_results = self._classify_with_rules(text, processed_text)
            
            # Combine with context if available
            if context:
                rule_results = self._incorporate_context(rule_results, context)
            
            # Determine primary intent
            primary_intent = self._determine_primary_intent(rule_results)
            
            # Detect urgency
            urgency = self._detect_urgency(text)
            
            # Extract key entities related to intent
            intent_entities = self._extract_intent_entities(text, primary_intent)
            
            return {
                "primary_intent": primary_intent,
                "intent_scores": rule_results,
                "confidence": rule_results.get(primary_intent, 0.0),
                "urgency": urgency,
                "intent_entities": intent_entities,
                "analysis_method": "rule_based",
                "context_used": context is not None
            }
            
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return {
                "primary_intent": "unknown",
                "intent_scores": {},
                "confidence": 0.0,
                "urgency": "unknown",
                "intent_entities": {},
                "error": str(e)
            }
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for intent classification."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.,!?-]', '', text)
        
        return text
    
    def _classify_with_rules(self, original_text: str, processed_text: str) -> Dict[str, float]:
        """Classify intent using rule-based approach."""
        intent_scores = {}
        
        for intent_name, patterns in self.intent_patterns.items():
            score = 0.0
            
            # Keyword matching
            keywords = patterns.get("keywords", [])
            keyword_matches = sum(1 for keyword in keywords if keyword in processed_text)
            if keywords:
                keyword_score = keyword_matches / len(keywords)
                score += keyword_score * 0.4
            
            # Phrase matching
            phrases = patterns.get("phrases", [])
            phrase_matches = sum(1 for phrase in phrases if phrase in processed_text)
            if phrases:
                phrase_score = phrase_matches / len(phrases)
                score += phrase_score * 0.5
            
            # Pattern matching
            regex_patterns = patterns.get("patterns", [])
            pattern_matches = sum(1 for pattern in regex_patterns if re.search(pattern, processed_text))
            if regex_patterns:
                pattern_score = pattern_matches / len(regex_patterns)
                score += pattern_score * 0.6
            
            # Apply confidence boost if any matches found
            if score > 0:
                confidence_boost = patterns.get("confidence_boost", 0.0)
                score = min(1.0, score + confidence_boost)
            
            intent_scores[intent_name] = score
        
        return intent_scores
    
    def _incorporate_context(
        self,
        intent_scores: Dict[str, float],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Incorporate context information into intent scores."""
        # Check for time-based context
        if context.get("time_mentioned"):
            intent_scores["appointment_booking"] += 0.1
        
        # Check for previous interaction context
        if context.get("previous_appointment"):
            intent_scores["follow_up"] += 0.15
        
        # Check for caller history
        caller_history = context.get("caller_history", {})
        if caller_history.get("recent_treatment"):
            intent_scores["follow_up"] += 0.1
        
        if caller_history.get("billing_issues"):
            intent_scores["billing_inquiry"] += 0.1
        
        # Ensure scores don't exceed 1.0
        for intent in intent_scores:
            intent_scores[intent] = min(1.0, intent_scores[intent])
        
        return intent_scores
    
    def _determine_primary_intent(self, intent_scores: Dict[str, float]) -> str:
        """Determine the primary intent from scores."""
        if not intent_scores:
            return "unknown"
        
        # Find intent with highest score
        primary_intent = max(intent_scores, key=intent_scores.get)
        
        # Only return if confidence is above threshold
        if intent_scores[primary_intent] >= 0.3:
            return primary_intent
        else:
            return "general_inquiry"  # Default to general inquiry for low confidence
    
    def _detect_urgency(self, text: str) -> str:
        """Detect urgency level from text."""
        text_lower = text.lower()
        
        # Check for high urgency indicators
        high_urgency_count = sum(1 for indicator in self.urgency_indicators["high"] 
                                if indicator in text_lower)
        
        # Check for medium urgency indicators
        medium_urgency_count = sum(1 for indicator in self.urgency_indicators["medium"] 
                                  if indicator in text_lower)
        
        # Check for low urgency indicators
        low_urgency_count = sum(1 for indicator in self.urgency_indicators["low"] 
                               if indicator in text_lower)
        
        # Determine urgency level
        if high_urgency_count > 0:
            return "high"
        elif medium_urgency_count > 0 and low_urgency_count == 0:
            return "medium"
        elif low_urgency_count > 0:
            return "low"
        else:
            return "medium"  # Default to medium if unclear
    
    def _extract_intent_entities(self, text: str, intent: str) -> Dict[str, Any]:
        """Extract entities relevant to the detected intent."""
        entities = {}
        
        if intent == "appointment_booking":
            entities.update(self._extract_scheduling_entities(text))
        elif intent == "emergency_dental":
            entities.update(self._extract_emergency_entities(text))
        elif intent == "insurance_inquiry":
            entities.update(self._extract_insurance_entities(text))
        elif intent == "treatment_inquiry":
            entities.update(self._extract_treatment_entities(text))
        
        return entities
    
    def _extract_scheduling_entities(self, text: str) -> Dict[str, Any]:
        """Extract scheduling-related entities."""
        entities = {}
        
        # Time expressions
        time_patterns = [
            r'\\b(?:1[0-2]|[1-9])(?::[0-5][0-9])?\\s*(?:am|pm|AM|PM)\\b',
            r'\\b(?:morning|afternoon|evening|noon)\\b'
        ]
        
        times = []
        for pattern in time_patterns:
            times.extend(re.findall(pattern, text))
        
        if times:
            entities["preferred_times"] = times
        
        # Day expressions
        day_pattern = r'\\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|next week|this week)\\b'
        days = re.findall(day_pattern, text, re.IGNORECASE)
        
        if days:
            entities["preferred_days"] = days
        
        return entities
    
    def _extract_emergency_entities(self, text: str) -> Dict[str, Any]:
        """Extract emergency-related entities."""
        entities = {}
        
        # Pain indicators
        pain_patterns = [
            r'(?:severe|terrible|unbearable|excruciating)\\s*pain',
            r'pain\\s*(?:level|scale)\\s*(?:[0-9]|ten)',
            r'(?:throbbing|sharp|dull)\\s*(?:pain|ache)'
        ]
        
        pain_descriptions = []
        for pattern in pain_patterns:
            pain_descriptions.extend(re.findall(pattern, text, re.IGNORECASE))
        
        if pain_descriptions:
            entities["pain_description"] = pain_descriptions
        
        # Symptoms
        symptom_keywords = ["bleeding", "swollen", "infected", "pus", "fever", "numbness"]
        symptoms = [symptom for symptom in symptom_keywords if symptom in text.lower()]
        
        if symptoms:
            entities["symptoms"] = symptoms
        
        return entities
    
    def _extract_insurance_entities(self, text: str) -> Dict[str, Any]:
        """Extract insurance-related entities."""
        entities = {}
        
        # Insurance providers
        providers = [
            "blue cross", "blue shield", "aetna", "cigna", "humana",
            "metlife", "delta dental", "guardian", "principal"
        ]
        
        found_providers = [provider for provider in providers if provider in text.lower()]
        if found_providers:
            entities["insurance_providers"] = found_providers
        
        # Coverage terms
        coverage_terms = ["copay", "deductible", "coverage", "benefits", "claim"]
        found_terms = [term for term in coverage_terms if term in text.lower()]
        if found_terms:
            entities["coverage_terms"] = found_terms
        
        return entities
    
    def _extract_treatment_entities(self, text: str) -> Dict[str, Any]:
        """Extract treatment-related entities."""
        entities = {}
        
        # Treatment types
        treatments = [
            "cleaning", "checkup", "filling", "crown", "root canal",
            "extraction", "whitening", "implant", "braces", "surgery"
        ]
        
        mentioned_treatments = [treatment for treatment in treatments if treatment in text.lower()]
        if mentioned_treatments:
            entities["treatments_mentioned"] = mentioned_treatments
        
        return entities
    
    def batch_classify(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Classify intents for multiple texts."""
        results = []
        
        for text in texts:
            result = self.classify_intent(text)
            results.append(result)
        
        return results
    
    def get_intent_patterns(self) -> Dict[str, Any]:
        """Get available intent patterns for inspection."""
        return {
            "intents": list(self.intent_patterns.keys()),
            "patterns": self.intent_patterns,
            "urgency_levels": list(self.urgency_indicators.keys())
        }
    
    def update_patterns(self, new_patterns: Dict[str, Any]) -> None:
        """Update intent patterns (for customization)."""
        for intent_name, pattern_data in new_patterns.items():
            if intent_name in self.intent_patterns:
                self.intent_patterns[intent_name].update(pattern_data)
            else:
                self.intent_patterns[intent_name] = pattern_data
        
        logger.info(f"Updated patterns for {len(new_patterns)} intents")