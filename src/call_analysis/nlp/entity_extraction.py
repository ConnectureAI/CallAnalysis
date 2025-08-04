"""
Named Entity Recognition and custom entity extraction.

This module provides sophisticated entity extraction capabilities including
standard NER, medical term extraction, and domain-specific entity recognition
for dental practice contexts.
"""

import logging
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass

import spacy
from spacy.matcher import Matcher, PhraseMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str


class EntityExtractor:
    """
    Advanced entity extraction using spaCy NER and custom patterns.
    
    Extracts standard entities (PERSON, ORG, LOC, etc.) as well as
    domain-specific entities relevant to dental practices.
    """
    
    def __init__(self):
        """Initialize entity extractor with models and patterns."""
        self._load_models()
        self._setup_custom_patterns()
        self._load_medical_vocabularies()
    
    def _load_models(self):
        """Load spaCy models and NLTK data."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            self.stop_words = set(stopwords.words('english'))
            logger.info("Entity extraction models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading entity extraction models: {e}")
            self.nlp = None
    
    def _setup_custom_patterns(self):
        """Set up custom entity patterns for dental practice domain."""
        if not self.nlp:
            return
        
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
        
        # Phone number patterns
        phone_patterns = [
            [{"SHAPE": "ddd"}, {"TEXT": "-"}, {"SHAPE": "ddd"}, {"TEXT": "-"}, {"SHAPE": "dddd"}],
            [{"TEXT": "("}, {"SHAPE": "ddd"}, {"TEXT": ")"}, {"SHAPE": "ddd"}, {"TEXT": "-"}, {"SHAPE": "dddd"}],
            [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}],
        ]
        
        for i, pattern in enumerate(phone_patterns):
            self.matcher.add(f"PHONE_{i}", [pattern])
        
        # Appointment time patterns
        time_patterns = [
            [{"SHAPE": "d"}, {"LOWER": {"IN": ["am", "pm"]}}],
            [{"SHAPE": "dd"}, {"LOWER": {"IN": ["am", "pm"]}}],
            [{"SHAPE": "d:dd"}, {"LOWER": {"IN": ["am", "pm"]}}],
            [{"SHAPE": "dd:dd"}, {"LOWER": {"IN": ["am", "pm"]}}],
        ]
        
        for i, pattern in enumerate(time_patterns):
            self.matcher.add(f"TIME_{i}", [pattern])
        
        # Insurance patterns
        insurance_patterns = [
            [{"LOWER": "blue"}, {"LOWER": "cross"}],
            [{"LOWER": "aetna"}],
            [{"LOWER": "cigna"}],
            [{"LOWER": "humana"}],
            [{"LOWER": "metlife"}],
            [{"LOWER": "delta"}, {"LOWER": "dental"}],
        ]
        
        for i, pattern in enumerate(insurance_patterns):
            self.matcher.add(f"INSURANCE_{i}", [pattern])
    
    def _load_medical_vocabularies(self):
        """Load medical and dental terminology."""
        # Dental procedures
        self.dental_procedures = {
            "cleaning", "checkup", "examination", "x-ray", "filling", "crown",
            "root canal", "extraction", "implant", "bridge", "dentures",
            "whitening", "scaling", "polishing", "fluoride treatment",
            "periodontal", "endodontic", "orthodontic", "oral surgery"
        }
        
        # Dental conditions
        self.dental_conditions = {
            "cavity", "caries", "gingivitis", "periodontitis", "plaque",
            "tartar", "abscess", "infection", "sensitivity", "toothache",
            "gum disease", "tooth decay", "bad breath", "dry mouth",
            "TMJ", "bruxism", "malocclusion"
        }
        
        # Symptoms
        self.symptoms = {
            "pain", "ache", "swelling", "bleeding", "sensitivity",
            "discomfort", "throbbing", "sharp pain", "dull ache",
            "tender", "sore", "inflammation", "irritation"
        }
        
        # Body parts (dental)
        self.dental_anatomy = {
            "tooth", "teeth", "molar", "incisor", "canine", "premolar",
            "gum", "gums", "jaw", "tongue", "mouth", "bite", "enamel",
            "dentin", "pulp", "nerve", "root", "crown"
        }
        
        # Combine all medical terms
        self.medical_terms = (
            self.dental_procedures | self.dental_conditions | 
            self.symptoms | self.dental_anatomy
        )
        
        # Create phrase patterns for medical terms
        if self.nlp:
            medical_patterns = [self.nlp(term) for term in self.medical_terms]
            self.phrase_matcher.add("MEDICAL_TERM", medical_patterns)
    
    def extract_all(self, text: str) -> Dict[str, List[Any]]:
        """
        Extract all types of entities from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing different types of extracted entities
        """
        if not self.nlp:
            return self._fallback_extraction(text)
        
        try:
            doc = self.nlp(text)
            
            # Standard NER entities
            standard_entities = self._extract_standard_entities(doc)
            
            # Custom pattern matches
            custom_entities = self._extract_custom_entities(doc)
            
            # Medical terms
            medical_entities = self._extract_medical_terms(doc)
            
            # Phone numbers using regex as backup
            phone_numbers = self._extract_phone_numbers_regex(text)
            
            # Email addresses
            email_addresses = self._extract_emails(text)
            
            return {
                "all_entities": (
                    standard_entities + custom_entities + 
                    medical_entities + phone_numbers + email_addresses
                ),
                "persons": [e.text for e in standard_entities if e.label == "PERSON"],
                "organizations": [e.text for e in standard_entities if e.label == "ORG"],
                "locations": [e.text for e in standard_entities if e.label in ["GPE", "LOC"]],
                "dates": [e.text for e in standard_entities if e.label == "DATE"],
                "times": [e.text for e in custom_entities if e.label.startswith("TIME")],
                "phone_numbers": [e.text for e in phone_numbers],
                "email_addresses": [e.text for e in email_addresses],
                "medical_terms": [e.text for e in medical_entities],
                "insurance_providers": [e.text for e in custom_entities if e.label.startswith("INSURANCE")],
            }
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return self._fallback_extraction(text)
    
    def _extract_standard_entities(self, doc) -> List[ExtractedEntity]:
        """Extract standard spaCy NER entities."""
        entities = []
        
        for ent in doc.ents:
            # Get surrounding context (5 words before and after)
            start_token = max(0, ent.start - 5)
            end_token = min(len(doc), ent.end + 5)
            context = doc[start_token:end_token].text
            
            entities.append(ExtractedEntity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence scores
                context=context
            ))
        
        return entities
    
    def _extract_custom_entities(self, doc) -> List[ExtractedEntity]:
        """Extract entities using custom patterns."""
        entities = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            # Get context
            start_token = max(0, start - 5)
            end_token = min(len(doc), end + 5)
            context = doc[start_token:end_token].text
            
            entities.append(ExtractedEntity(
                text=span.text,
                label=label,
                start=span.start_char,
                end=span.end_char,
                confidence=0.9,
                context=context
            ))
        
        return entities
    
    def _extract_medical_terms(self, doc) -> List[ExtractedEntity]:
        """Extract medical and dental terms."""
        entities = []
        matches = self.phrase_matcher(doc)
        
        for match_id, start, end in matches:
            span = doc[start:end]
            
            # Get context
            start_token = max(0, start - 5)
            end_token = min(len(doc), end + 5)
            context = doc[start_token:end_token].text
            
            entities.append(ExtractedEntity(
                text=span.text,
                label="MEDICAL_TERM",
                start=span.start_char,
                end=span.end_char,
                confidence=0.8,
                context=context
            ))
        
        return entities
    
    def _extract_phone_numbers_regex(self, text: str) -> List[ExtractedEntity]:
        """Extract phone numbers using regex patterns."""
        entities = []
        
        # Common phone number patterns
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            r'\b\d{3}\s\d{3}\s\d{4}\b',
        ]
        
        for pattern in phone_patterns:
            for match in re.finditer(pattern, text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label="PHONE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    context=self._get_context(text, match.start(), match.end())
                ))
        
        return entities
    
    def _extract_emails(self, text: str) -> List[ExtractedEntity]:
        """Extract email addresses."""
        entities = []
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        for match in re.finditer(email_pattern, text):
            entities.append(ExtractedEntity(
                text=match.group(),
                label="EMAIL",
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                context=self._get_context(text, match.start(), match.end())
            ))
        
        return entities
    
    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get surrounding context for an entity."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]
    
    def _fallback_extraction(self, text: str) -> Dict[str, List[Any]]:
        """Fallback entity extraction using regex patterns."""
        logger.warning("Using fallback entity extraction")
        
        entities = []
        
        # Extract phone numbers
        phone_entities = self._extract_phone_numbers_regex(text)
        entities.extend(phone_entities)
        
        # Extract emails
        email_entities = self._extract_emails(text)
        entities.extend(email_entities)
        
        # Simple medical term matching
        text_lower = text.lower()
        for term in self.medical_terms:
            if term in text_lower:
                start = text_lower.find(term)
                end = start + len(term)
                entities.append(ExtractedEntity(
                    text=term,
                    label="MEDICAL_TERM",
                    start=start,
                    end=end,
                    confidence=0.5,
                    context=self._get_context(text, start, end)
                ))
        
        return {
            "all_entities": entities,
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
            "times": [],
            "phone_numbers": [e.text for e in phone_entities],
            "email_addresses": [e.text for e in email_entities],
            "medical_terms": [e.text for e in entities if e.label == "MEDICAL_TERM"],
            "insurance_providers": [],
        }
    
    def extract_appointments_info(self, text: str) -> Dict[str, Any]:
        """
        Extract appointment-specific information.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary containing appointment-related entities
        """
        # Time expressions
        time_patterns = [
            r'\b(?:1[0-2]|[1-9])(?::[0-5][0-9])?\s*(?:am|pm|AM|PM)\b',
            r'\b(?:morning|afternoon|evening|noon)\b',
        ]
        
        # Day expressions
        day_patterns = [
            r'\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(?:today|tomorrow|next week|this week)\b',
        ]
        
        # Date expressions
        date_patterns = [
            r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b',
            r'\b\d{1,2}\/\d{1,2}\/\d{2,4}\b',
        ]
        
        times = []
        days = []
        dates = []
        
        text_lower = text.lower()
        
        for pattern in time_patterns:
            times.extend([m.group() for m in re.finditer(pattern, text, re.IGNORECASE)])
        
        for pattern in day_patterns:
            days.extend([m.group() for m in re.finditer(pattern, text_lower)])
        
        for pattern in date_patterns:
            dates.extend([m.group() for m in re.finditer(pattern, text, re.IGNORECASE)])
        
        return {
            "times": times,
            "days": days,
            "dates": dates,
            "has_scheduling_intent": any([
                keyword in text_lower for keyword in 
                ["schedule", "book", "appointment", "available", "when can"]
            ])
        }
    
    def extract_insurance_info(self, text: str) -> Dict[str, Any]:
        """Extract insurance-related information."""
        insurance_providers = [
            "blue cross", "blue shield", "aetna", "cigna", "humana",
            "metlife", "delta dental", "guardian", "principal",
            "united healthcare", "anthem"
        ]
        
        insurance_terms = [
            "insurance", "coverage", "benefits", "copay", "deductible",
            "claim", "pre-authorization", "in-network", "out-of-network"
        ]
        
        text_lower = text.lower()
        
        found_providers = [provider for provider in insurance_providers if provider in text_lower]
        found_terms = [term for term in insurance_terms if term in text_lower]
        
        return {
            "insurance_providers": found_providers,
            "insurance_terms": found_terms,
            "has_insurance_inquiry": len(found_terms) > 0 or len(found_providers) > 0
        }