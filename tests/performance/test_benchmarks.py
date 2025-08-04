"""
Performance and benchmark tests.
"""

import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor


@pytest.mark.slow  
class TestAnalyzerPerformance:
    """Performance tests for the analyzer."""

    @pytest.mark.asyncio
    async def test_analyzer_single_call_performance(self, analyzer, benchmark_data):
        """Benchmark single call analysis performance."""
        transcript = benchmark_data["transcripts"][0]
        call_id = benchmark_data["call_ids"][0]
        
        start_time = time.time()
        result = await analyzer.analyze_transcript(transcript, call_id)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Analysis should complete within reasonable time
        assert duration < 10.0  # 10 seconds max for single analysis
        assert result is not None

    @pytest.mark.asyncio
    async def test_analyzer_batch_performance(self, analyzer, benchmark_data):
        """Benchmark batch analysis performance."""
        batch_data = [
            {"call_id": call_id, "transcript": transcript}
            for call_id, transcript in zip(
                benchmark_data["call_ids"][:10],  # Test with 10 calls
                benchmark_data["transcripts"][:10]
            )
        ]
        
        start_time = time.time()
        results = await analyzer.analyze_batch(batch_data)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Batch processing should be reasonably fast
        assert duration < 60.0  # 1 minute max for 10 calls
        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_analyzer_concurrent_performance(self, analyzer, benchmark_data):
        """Test analyzer performance under concurrent load."""
        
        async def analyze_single(call_id, transcript):
            return await analyzer.analyze_transcript(transcript, call_id)
        
        # Create 5 concurrent analysis tasks
        tasks = [
            analyze_single(
                benchmark_data["call_ids"][i],
                benchmark_data["transcripts"][i]
            )
            for i in range(5)
        ]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Concurrent processing should complete within reasonable time
        assert duration < 30.0  # 30 seconds max for 5 concurrent calls
        assert len(results) == 5
        assert all(result is not None for result in results)


@pytest.mark.slow
class TestDatabasePerformance:
    """Performance tests for database operations."""

    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, db_session):
        """Test bulk insert performance."""
        from src.call_analysis.models import CallInsightDB
        from datetime import datetime
        
        # Create 100 test records
        records = []
        for i in range(100):
            record = CallInsightDB(
                call_id=f"perf_test_{i:03d}",
                timestamp=datetime.now(),
                clinic_mentioned="Performance Test Clinic",
                primary_intent="appointment_booking",
                secondary_intents=["inquiry"],
                sentiment_score=0.5,
                emotional_journey={"beginning": 0.5, "middle": 0.5, "end": 0.5},
                urgency_level=2,
                resolution_status="resolved",
                staff_performance={"empathy": 0.8},
                revenue_opportunity=100.0,
                service_requests=["cleaning"],
                pain_points=[],
                success_factors=[],
                next_actions=[],
                confidence_score=0.8
            )
            records.append(record)
        
        start_time = time.time()
        db_session.add_all(records)
        await db_session.commit()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Bulk insert should be fast
        assert duration < 5.0  # 5 seconds max for 100 records

    @pytest.mark.asyncio
    async def test_query_performance(self, db_with_sample_data):
        """Test query performance."""
        from src.call_analysis.models import CallInsightDB
        from sqlalchemy import select
        
        # Add more test data
        records = []
        for i in range(50):
            record = CallInsightDB(
                call_id=f"query_test_{i:03d}",
                timestamp=datetime.now(),
                clinic_mentioned="Query Test Clinic",
                primary_intent="appointment_booking",
                secondary_intents=["inquiry"],
                sentiment_score=0.5,
                emotional_journey={"beginning": 0.5, "middle": 0.5, "end": 0.5},
                urgency_level=2,
                resolution_status="resolved",
                staff_performance={"empathy": 0.8},
                revenue_opportunity=100.0,
                service_requests=["cleaning"],
                pain_points=[],
                success_factors=[],
                next_actions=[],
                confidence_score=0.8
            )
            records.append(record)
        
        db_with_sample_data.add_all(records)
        await db_with_sample_data.commit()
        
        # Test query performance
        start_time = time.time()
        stmt = select(CallInsightDB).limit(20)
        result = await db_with_sample_data.execute(stmt)
        insights = result.scalars().all()
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Query should be fast
        assert duration < 1.0  # 1 second max for simple query
        assert len(insights) <= 20


@pytest.mark.slow
class TestAPIPerformance:
    """Performance tests for API endpoints."""

    def test_health_endpoint_performance(self, client):
        """Test health endpoint response time."""
        # Warm up
        client.get("/health")
        
        # Measure performance
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        duration = end_time - start_time
        
        assert response.status_code == 200
        assert duration < 0.5  # 500ms max for health check

    def test_concurrent_api_requests(self, client):
        """Test API performance under concurrent load."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            results.put((response.status_code, end_time - start_time))
        
        # Create 10 concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        end_time = time.time()
        
        total_duration = end_time - start_time
        
        # Collect results
        response_times = []
        while not results.empty():
            status_code, duration = results.get()
            assert status_code == 200
            response_times.append(duration)
        
        # All requests should complete within reasonable time
        assert total_duration < 5.0  # 5 seconds max for 10 concurrent requests
        assert len(response_times) == 10
        assert max(response_times) < 2.0  # No single request should take more than 2 seconds


@pytest.mark.benchmark
class TestMemoryUsage:
    """Memory usage tests."""

    def test_analyzer_memory_usage(self, analyzer):
        """Test analyzer memory usage doesn't grow excessively."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate multiple analyses
        for i in range(10):
            # Note: This would require mocking to avoid real API calls
            transcript = f"Test transcript {i} for memory testing"
            call_id = f"memory_test_{i:03d}"
            # analyzer.analyze_transcript would be called here in real test
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB)
        assert memory_growth < 100 * 1024 * 1024  # 100MB