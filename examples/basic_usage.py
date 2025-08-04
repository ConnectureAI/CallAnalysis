#!/usr/bin/env python3
"""
Basic usage example for the Call Analysis System.

This example demonstrates how to use the AI agent system
and advanced NLP capabilities.
"""

import asyncio
import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def main():
    """Main example function."""
    print("🤖 Call Analysis System - Basic Usage Example")
    print("=" * 50)
    
    try:
        # Import the necessary components
        from call_analysis.agents import AgentManager
        from call_analysis.nlp import (
            IntentClassifier,
            EntityExtractor, 
            TopicModelingEngine
        )
        
        print("✅ Successfully imported Call Analysis components")
        
        # Example 1: Direct NLP Analysis
        print("\n📝 Example 1: Direct NLP Analysis")
        print("-" * 30)
        
        # Sample call transcript
        transcript = """
        Hi, this is Sarah calling. I'm having severe tooth pain and need to see 
        Dr. Smith as soon as possible. My insurance is Blue Cross Blue Shield 
        and my phone number is 555-123-4567. Can you fit me in today or tomorrow?
        This is really urgent - I can barely eat anything.
        """
        
        print(f"Analyzing transcript: {transcript.strip()[:100]}...")
        
        # Initialize NLP components
        intent_classifier = IntentClassifier()
        entity_extractor = EntityExtractor()
        topic_engine = TopicModelingEngine()
        
        # Perform analysis
        intent_result = intent_classifier.classify_intent(transcript)
        entities = entity_extractor.extract_all(transcript)
        topics = topic_engine.analyze_topics(transcript)
        
        # Display results
        print(f"\n🎯 Intent: {intent_result['primary_intent']} (confidence: {intent_result['confidence']:.2f})")
        print(f"⚡ Urgency: {intent_result['urgency']}")
        
        print(f"\n🏷️  Entities Found:")
        if entities['phone_numbers']:
            print(f"  📞 Phone: {entities['phone_numbers']}")
        if entities['insurance_providers']:
            print(f"  🏥 Insurance: {entities['insurance_providers']}")
        if entities['medical_terms']:
            print(f"  🩺 Medical Terms: {entities['medical_terms']}")
        
        print(f"\n📊 Topics:")
        print(f"  🎯 Dominant Topic: {topics['dominant_topic']}")
        if topics['key_themes']:
            print(f"  🔑 Key Themes: {', '.join(topics['key_themes'])}")
        
        # Example 2: Agent System (if dependencies are available)
        print("\n\n🤖 Example 2: AI Agent System")
        print("-" * 30)
        
        try:
            # Initialize agent manager
            manager = AgentManager()
            print("✅ Agent Manager initialized")
            
            # Note: In a real deployment, you would:
            # 1. await manager.initialize()
            # 2. Send messages to agents
            # 3. Process results 
            # 4. await manager.shutdown()
            
            print("📋 Available Agents:")
            print("  • Call Acquisition Agent - Monitors call sources")
            print("  • Analysis Orchestrator - Coordinates NLP analysis")
            print("  • Monitoring Agent - Tracks system health")
            print("  • Workflow Agent - Manages complex workflows")
            
            print("\n💡 To start the agent system:")
            print("   python -m call_analysis.main agents start")
            
        except Exception as e:
            print(f"⚠️  Agent system requires full dependencies: {e}")
            print("   This is expected in a minimal installation")
        
        # Example 3: Workflow Definition
        print("\n\n⚙️  Example 3: Workflow Definition")
        print("-" * 30)
        
        workflow_example = {
            "name": "Emergency Call Processing",
            "description": "Handle urgent dental emergencies",
            "steps": [
                {
                    "name": "Detect Urgency",
                    "agent": "analysis_orchestrator",
                    "action": "classify_intent",
                    "parameters": {"text": "{input.transcript}"}
                },
                {
                    "name": "Extract Contact Info", 
                    "agent": "analysis_orchestrator",
                    "action": "extract_entities",
                    "parameters": {"text": "{input.transcript}"}
                },
                {
                    "name": "Generate Alert",
                    "agent": "monitoring_agent",
                    "action": "create_alert",
                    "conditions": {
                        "require": ["step.detect_urgency.result.urgency == 'high'"]
                    }
                }
            ]
        }
        
        print("📋 Example Workflow Structure:")
        for i, step in enumerate(workflow_example["steps"], 1):
            print(f"  {i}. {step['name']} ({step['agent']})")
        
        print("\n\n🎉 Example Complete!")
        print("\nNext Steps:")
        print("1. Install full dependencies: pip install -r requirements.txt")
        print("2. Download NLP models: python -m spacy download en_core_web_sm")
        print("3. Start the system: python -m call_analysis.main agents start")
        print("4. Check status: python -m call_analysis.main agents status")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nThis is expected if dependencies are not installed.")
        print("The system structure is correct and ready for deployment.")
        print("\nTo install dependencies:")
        print("  pip install -r requirements.txt")
        print("  python -m spacy download en_core_web_sm")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)