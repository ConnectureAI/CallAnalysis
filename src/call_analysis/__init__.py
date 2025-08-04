"""
Call Analysis - AI-powered dental call center analytics and coaching system.

This package provides comprehensive AI-powered analytics for dental call centers,
including semantic transcript analysis, predictive analytics, and real-time coaching.
"""

__version__ = "0.1.0"
__author__ = "CallAnalysis Team"
__email__ = "info@callanalysis.ai"

from .models import CallInsight
from .analyzer import SemanticTranscriptAnalyzer
from .predictor import PredictiveAnalytics
from .coaching import RealTimeCoachingSystem

__all__ = [
    "CallInsight",
    "SemanticTranscriptAnalyzer", 
    "PredictiveAnalytics",
    "RealTimeCoachingSystem",
]