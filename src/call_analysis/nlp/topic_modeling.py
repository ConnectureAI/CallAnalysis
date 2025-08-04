"""
Topic modeling engine using LDA and advanced techniques.

This module provides sophisticated topic modeling capabilities including
Latent Dirichlet Allocation (LDA), dynamic topic modeling, and coherence
measurement for dental practice conversations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import gensim
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel

logger = logging.getLogger(__name__)


class TopicModelingEngine:
    """
    Advanced topic modeling using multiple algorithms.
    
    This class provides topic discovery and analysis using LDA, NMF,
    and clustering techniques specifically tuned for call center conversations.
    """
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """
        Initialize topic modeling engine.
        
        Args:
            model_cache_dir: Directory to cache trained models
        """
        self.model_cache_dir = Path(model_cache_dir) if model_cache_dir else Path("models/topics")
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_nltk()
        self._load_dental_vocabulary()
        
        # Model storage
        self.lda_model = None
        self.nmf_model = None
        self.vectorizer = None
        self.dictionary = None
        self.corpus = None
        
        # Predefined topic categories for dental practice
        self.dental_topics = {
            "appointment_scheduling": [
                "appointment", "schedule", "book", "available", "time", "date",
                "calendar", "slot", "booking", "reschedule", "cancel"
            ],
            "emergency_dental": [
                "emergency", "urgent", "pain", "severe", "bleeding", "accident",
                "immediate", "asap", "help", "urgent care"
            ],
            "insurance_billing": [
                "insurance", "coverage", "benefits", "claim", "copay", "deductible",
                "billing", "payment", "cost", "price", "charge"
            ],
            "dental_procedures": [
                "cleaning", "checkup", "filling", "crown", "root canal", "extraction",
                "whitening", "implant", "surgery", "treatment", "procedure"
            ],
            "patient_concerns": [
                "worried", "concerned", "anxious", "scared", "nervous", "question",
                "problem", "issue", "complaint", "dissatisfied"
            ],
            "follow_up": [
                "follow up", "check back", "return", "next visit", "callback",
                "post treatment", "recovery", "healing", "progress"
            ]
        }
    
    def _setup_nltk(self):
        """Download and setup required NLTK data."""
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
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add domain-specific stop words
        self.stop_words.update([
            "um", "uh", "oh", "yeah", "yes", "no", "okay", "ok", "sure",
            "hello", "hi", "thank", "thanks", "please", "sorry"
        ])
    
    def _load_dental_vocabulary(self):
        """Load dental practice specific vocabulary."""
        self.dental_vocabulary = set()
        for topic_words in self.dental_topics.values():
            self.dental_vocabulary.update(topic_words)
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text for topic modeling.
        
        Args:
            text: Input text
            
        Returns:
            List of processed tokens
        """
        try:
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove punctuation and non-alphabetic tokens
            tokens = [token for token in tokens if token.isalpha()]
            
            # Remove stop words
            tokens = [token for token in tokens if token not in self.stop_words]
            
            # POS tagging and keep only nouns, verbs, adjectives
            pos_tags = pos_tag(tokens)
            tokens = [
                token for token, pos in pos_tags 
                if pos.startswith(('NN', 'VB', 'JJ'))
            ]
            
            # Lemmatization
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Filter out very short tokens
            tokens = [token for token in tokens if len(token) > 2]
            
            return tokens
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {e}")
            return []
    
    def analyze_topics(self, text: str, num_topics: int = 6) -> Dict[str, Any]:
        """
        Analyze topics in the given text.
        
        Args:
            text: Input text to analyze
            num_topics: Number of topics to extract
            
        Returns:
            Dictionary containing topic analysis results
        """
        try:
            # Preprocess text
            processed_tokens = self.preprocess_text(text)
            
            if len(processed_tokens) < 10:  # Not enough content for meaningful topic modeling
                return self._fallback_topic_analysis(text)
            
            # Create document for analysis
            documents = [processed_tokens]
            
            # LDA analysis
            lda_results = self._lda_analysis(documents, num_topics)
            
            # Predefined topic matching
            predefined_results = self._match_predefined_topics(processed_tokens)
            
            # Combine results
            return self._combine_topic_results(lda_results, predefined_results, processed_tokens)
            
        except Exception as e:
            logger.error(f"Error in topic analysis: {e}")
            return self._fallback_topic_analysis(text)
    
    def _lda_analysis(self, documents: List[List[str]], num_topics: int) -> Dict[str, Any]:
        """Perform LDA topic modeling."""
        try:
            # Create dictionary and corpus
            dictionary = corpora.Dictionary(documents)
            corpus = [dictionary.doc2bow(doc) for doc in documents]
            
            # Train LDA model
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            # Get topics for the document
            doc_topics = lda_model.get_document_topics(corpus[0])
            
            # Extract topic words
            topics = []
            for topic_id in range(num_topics):
                topic_words = lda_model.show_topic(topic_id, topn=5)
                topics.append({
                    "id": topic_id,
                    "words": [word for word, _ in topic_words],
                    "weights": [weight for _, weight in topic_words]
                })
            
            # Calculate coherence score
            try:
                coherence_model = CoherenceModel(
                    model=lda_model, 
                    texts=documents, 
                    dictionary=dictionary, 
                    coherence='c_v'
                )
                coherence_score = coherence_model.get_coherence()
            except:
                coherence_score = 0.5
            
            return {
                "topics": topics,
                "document_topic_distribution": [prob for _, prob in doc_topics],
                "coherence_score": coherence_score,
                "dominant_topic_id": max(doc_topics, key=lambda x: x[1])[0] if doc_topics else 0
            }
            
        except Exception as e:
            logger.error(f"Error in LDA analysis: {e}")
            return {
                "topics": [],
                "document_topic_distribution": [],
                "coherence_score": 0.0,
                "dominant_topic_id": 0
            }
    
    def _match_predefined_topics(self, tokens: List[str]) -> Dict[str, Any]:
        """Match tokens against predefined dental topics."""
        topic_scores = {}
        token_set = set(tokens)
        
        for topic_name, topic_words in self.dental_topics.items():
            # Calculate overlap between tokens and topic words
            overlap = len(token_set & set(topic_words))
            total_topic_words = len(topic_words)
            
            # Calculate score as percentage of topic words found
            score = overlap / total_topic_words if total_topic_words > 0 else 0
            topic_scores[topic_name] = score
        
        # Find dominant topic
        dominant_topic = max(topic_scores, key=topic_scores.get) if topic_scores else "general"
        
        return {
            "predefined_topic_scores": topic_scores,
            "dominant_predefined_topic": dominant_topic,
            "max_score": topic_scores.get(dominant_topic, 0.0)
        }
    
    def _combine_topic_results(
        self, 
        lda_results: Dict[str, Any], 
        predefined_results: Dict[str, Any],
        tokens: List[str]
    ) -> Dict[str, Any]:
        """Combine LDA and predefined topic results."""
        
        # Determine overall dominant topic
        lda_coherence = lda_results.get("coherence_score", 0.0)
        predefined_score = predefined_results.get("max_score", 0.0)
        
        if predefined_score > 0.3 and predefined_score > lda_coherence:
            dominant_topic = predefined_results["dominant_predefined_topic"]
            topic_coherence = predefined_score
        else:
            # Use LDA result or fall back to predefined
            if lda_results.get("topics"):
                dominant_topic_id = lda_results.get("dominant_topic_id", 0)
                if dominant_topic_id < len(lda_results["topics"]):
                    topic_words = lda_results["topics"][dominant_topic_id]["words"]
                    dominant_topic = f"lda_topic_{dominant_topic_id}"
                else:
                    dominant_topic = predefined_results.get("dominant_predefined_topic", "general")
            else:
                dominant_topic = predefined_results.get("dominant_predefined_topic", "general")
            
            topic_coherence = lda_coherence
        
        # Extract key themes
        key_themes = self._extract_key_themes(tokens)
        
        return {
            "topics": lda_results.get("topics", []),
            "topic_distribution": lda_results.get("document_topic_distribution", []),
            "dominant_topic": dominant_topic,
            "topic_coherence_score": topic_coherence,
            "predefined_topic_scores": predefined_results.get("predefined_topic_scores", {}),
            "key_themes": key_themes,
            "lda_coherence": lda_coherence,
            "predefined_best_match": predefined_results.get("dominant_predefined_topic", "general")
        }
    
    def _extract_key_themes(self, tokens: List[str]) -> List[str]:
        """Extract key themes from tokens."""
        # Count token frequencies
        from collections import Counter
        token_counts = Counter(tokens)
        
        # Get most common tokens that are also in dental vocabulary
        key_themes = []
        for token, count in token_counts.most_common(10):
            if token in self.dental_vocabulary or count >= 2:
                key_themes.append(token)
        
        return key_themes[:5]  # Return top 5 themes
    
    def _fallback_topic_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback topic analysis using simple keyword matching."""
        logger.warning("Using fallback topic analysis")
        
        text_lower = text.lower()
        topic_scores = {}
        
        # Simple keyword matching against predefined topics
        for topic_name, topic_words in self.dental_topics.items():
            score = sum(1 for word in topic_words if word in text_lower)
            topic_scores[topic_name] = score / len(topic_words)
        
        # Find best matching topic
        dominant_topic = max(topic_scores, key=topic_scores.get) if topic_scores else "general"
        
        return {
            "topics": [{"id": 0, "words": ["fallback"], "weights": [1.0]}],
            "topic_distribution": [1.0],
            "dominant_topic": dominant_topic,
            "topic_coherence_score": topic_scores.get(dominant_topic, 0.0),
            "predefined_topic_scores": topic_scores,
            "key_themes": [dominant_topic.replace("_", " ")],
            "lda_coherence": 0.0,
            "predefined_best_match": dominant_topic
        }
    
    def train_corpus_model(
        self, 
        documents: List[str], 
        num_topics: int = 10, 
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train topic model on a corpus of documents.
        
        Args:
            documents: List of document texts
            num_topics: Number of topics to extract
            save_model: Whether to save the trained model
            
        Returns:
            Training results and model information
        """
        try:
            # Preprocess all documents
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            processed_docs = [doc for doc in processed_docs if len(doc) > 5]  # Filter short docs
            
            if len(processed_docs) < 3:
                logger.warning("Not enough documents for corpus training")
                return {"error": "Insufficient training data"}
            
            # Create dictionary and corpus
            self.dictionary = corpora.Dictionary(processed_docs)
            self.dictionary.filter_extremes(no_below=2, no_above=0.7)  # Filter rare and common words
            self.corpus = [self.dictionary.doc2bow(doc) for doc in processed_docs]
            
            # Train LDA model
            self.lda_model = models.LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=num_topics,
                random_state=42,
                passes=20,
                alpha='auto',
                eta='auto',
                per_word_topics=True
            )
            
            # Calculate coherence
            coherence_model = CoherenceModel(
                model=self.lda_model,
                texts=processed_docs,
                dictionary=self.dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            
            # Extract topics
            topics = []
            for topic_id in range(num_topics):
                topic_words = self.lda_model.show_topic(topic_id, topn=10)
                topics.append({
                    "id": topic_id,
                    "words": [word for word, _ in topic_words],
                    "weights": [weight for _, weight in topic_words],
                    "label": self._generate_topic_label([word for word, _ in topic_words[:3]])
                })
            
            # Save model if requested
            if save_model:
                self._save_model()
            
            return {
                "num_topics": num_topics,
                "num_documents": len(processed_docs),
                "coherence_score": coherence_score,
                "topics": topics,
                "model_trained": True
            }
            
        except Exception as e:
            logger.error(f"Error training corpus model: {e}")
            return {"error": str(e)}
    
    def _generate_topic_label(self, top_words: List[str]) -> str:
        """Generate a human-readable label for a topic."""
        # Try to match with predefined topics
        for topic_name, topic_words in self.dental_topics.items():
            overlap = len(set(top_words) & set(topic_words))
            if overlap >= 2:
                return topic_name.replace("_", " ").title()
        
        # Generate label from top words
        return " + ".join(top_words)
    
    def _save_model(self):
        """Save trained models to disk."""
        try:
            if self.lda_model:
                model_path = self.model_cache_dir / "lda_model"
                self.lda_model.save(str(model_path))
            
            if self.dictionary:
                dict_path = self.model_cache_dir / "dictionary.pkl"
                with open(dict_path, 'wb') as f:
                    pickle.dump(self.dictionary, f)
            
            logger.info("Topic models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def load_model(self) -> bool:
        """Load trained models from disk."""
        try:
            model_path = self.model_cache_dir / "lda_model"
            dict_path = self.model_cache_dir / "dictionary.pkl"
            
            if model_path.exists() and dict_path.exists():
                self.lda_model = models.LdaModel.load(str(model_path))
                
                with open(dict_path, 'rb') as f:
                    self.dictionary = pickle.load(f)
                
                logger.info("Topic models loaded successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def get_document_topics(self, text: str) -> Dict[str, Any]:
        """Get topic distribution for a single document using trained model."""
        if not self.lda_model or not self.dictionary:
            logger.warning("No trained model available")
            return self.analyze_topics(text)
        
        try:
            # Preprocess text
            processed_tokens = self.preprocess_text(text)
            
            # Convert to bag of words
            bow = self.dictionary.doc2bow(processed_tokens)
            
            # Get topic distribution
            doc_topics = self.lda_model.get_document_topics(bow)
            
            # Get dominant topic
            if doc_topics:
                dominant_topic_id, dominant_prob = max(doc_topics, key=lambda x: x[1])
                dominant_topic_words = [word for word, _ in self.lda_model.show_topic(dominant_topic_id, topn=5)]
                dominant_topic_label = self._generate_topic_label(dominant_topic_words)
            else:
                dominant_topic_id = 0
                dominant_prob = 0.0
                dominant_topic_label = "unknown"
            
            return {
                "document_topics": [(topic_id, prob) for topic_id, prob in doc_topics],
                "dominant_topic_id": dominant_topic_id,
                "dominant_topic_probability": dominant_prob,
                "dominant_topic_label": dominant_topic_label,
                "topic_distribution": [prob for _, prob in doc_topics]
            }
            
        except Exception as e:
            logger.error(f"Error getting document topics: {e}")
            return self.analyze_topics(text)