"""
Tests for the semantic transcript analyzer.
"""

import pytest
from unittest.mock import AsyncMock, patch
import json

from src.call_analysis.analyzer import SemanticTranscriptAnalyzer
from src.call_analysis.models import CallInsight


class TestSemanticTranscriptAnalyzer:
    """Test suite for SemanticTranscriptAnalyzer."""

    @pytest.mark.asyncio
    async def test_analyze_transcript_success(self, analyzer, sample_transcript):
        """Test successful transcript analysis."""
        # Test the analyzer with a sample transcript
        result = await analyzer.analyze_transcript(sample_transcript, "test_call_001")
        
        # Verify the result is a CallInsight object
        assert isinstance(result, CallInsight)
        assert result.call_id == "test_call_001"
        assert result.primary_intent is not None
        assert -1.0 <= result.sentiment_score <= 1.0
        assert 1 <= result.urgency_level <= 5
        assert 0.0 <= result.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_transcript_with_cache(self, analyzer, sample_transcript):
        """Test that analyzer uses cache for repeated calls."""
        call_id = "test_call_cache"
        
        # First call
        result1 = await analyzer.analyze_transcript(sample_transcript, call_id)
        
        # Second call should use cache
        result2 = await analyzer.analyze_transcript(sample_transcript, call_id)
        
        # Results should be identical (from cache)
        assert result1.call_id == result2.call_id
        assert result1.sentiment_score == result2.sentiment_score
        assert result1.primary_intent == result2.primary_intent

    @pytest.mark.asyncio
    async def test_analyze_transcript_api_error(self, analyzer, sample_transcript):
        """Test handling of API errors with fallback."""
        # Mock API to raise an exception
        analyzer.client.chat.completions.create.side_effect = Exception("API Error")
        
        # Should still return a CallInsight (fallback)
        result = await analyzer.analyze_transcript(sample_transcript, "test_call_error")
        
        assert isinstance(result, CallInsight)
        assert result.call_id == "test_call_error"
        assert result.confidence_score <= 0.5  # Low confidence for fallback

    @pytest.mark.asyncio
    async def test_analyze_batch(self, analyzer):
        """Test batch analysis functionality."""
        batch_data = [
            {"call_id": "call_001", "transcript": "Hello, I need an appointment"},
            {"call_id": "call_002", "transcript": "I have a dental emergency"},
            {"call_id": "call_003", "transcript": "What are your office hours?"}
        ]
        
        results = await analyzer.analyze_batch(batch_data)
        
        assert len(results) == 3
        assert all(isinstance(result, CallInsight) for result in results)
        assert [result.call_id for result in results] == ["call_001", "call_002", "call_003"]

    @pytest.mark.asyncio
    async def test_validate_api_key_success(self, analyzer):
        """Test API key validation success."""
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.choices = [AsyncMock()]
        analyzer.client.chat.completions.create.return_value = mock_response
        
        is_valid = await analyzer.validate_api_key()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_validate_api_key_failure(self, analyzer):
        """Test API key validation failure."""
        # Mock API error
        analyzer.client.chat.completions.create.side_effect = Exception("Invalid API key")
        
        is_valid = await analyzer.validate_api_key()
        assert is_valid is False

    def test_intelligent_chunking(self, analyzer):
        """Test intelligent chunking of long transcripts."""
        # Create a long transcript
        long_transcript = "This is a test transcript. " * 1000
        
        # Mock encoding to return many tokens
        with patch.object(analyzer.encoding, 'encode', return_value=list(range(5000))):
            with patch.object(analyzer.encoding, 'decode', side_effect=lambda x: f"chunk_{len(x)}"):
                result = analyzer._intelligent_chunking(long_transcript, 1000)
                
                # Should contain chunked content with middle omitted
                assert "middle of conversation omitted" in result

    def test_create_fallback_insight(self, analyzer, sample_transcript):
        """Test fallback insight creation."""
        result = analyzer._create_fallback_insight("test_call", sample_transcript)
        
        assert isinstance(result, CallInsight)
        assert result.call_id == "test_call"
        assert result.confidence_score <= 0.5
        assert result.primary_intent is not None

    def test_cache_management(self, analyzer):
        """Test cache management functionality."""
        # Add some items to cache
        analyzer.analysis_cache["key1"] = "value1"
        analyzer.analysis_cache["key2"] = "value2"
        
        # Test cache stats
        stats = analyzer.get_cache_stats()
        assert stats["cache_size"] == 2
        assert "key1" in stats["cache_keys"]
        assert "key2" in stats["cache_keys"]
        
        # Test cache clearing
        analyzer.clear_cache()
        assert len(analyzer.analysis_cache) == 0


@pytest.mark.integration 
class TestSemanticTranscriptAnalyzerIntegration:
    """Integration tests for SemanticTranscriptAnalyzer."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not pytest.importorskip("openai"), reason="OpenAI not installed")
    async def test_real_openai_api(self, sample_transcript):
        """Test with real OpenAI API (requires valid API key)."""
        import os
        
        api_key = os.getenv("OPENAI_API_KEY_REAL")
        if not api_key or api_key.startswith("sk-test"):
            pytest.skip("Real OpenAI API key not available")
        
        analyzer = SemanticTranscriptAnalyzer(api_key=api_key)
        result = await analyzer.analyze_transcript(sample_transcript, "integration_test")
        
        assert isinstance(result, CallInsight)
        assert result.confidence_score > 0.5  # Should have high confidence with real API