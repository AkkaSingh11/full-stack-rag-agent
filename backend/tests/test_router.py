#!/usr/bin/env python3
"""
Quick test script to verify the orchestrator router pattern.
Tests both conversational and research paths.
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

load_dotenv()

from langchain_core.messages import HumanMessage
from agent.graph import graph

def test_router():
    """Test the router with different query types"""

    test_cases = [
        {
            "query": "Hi",
            "expected_route": "conversational",
            "description": "Simple greeting"
        },
        {
            "query": "Hello, how are you?",
            "expected_route": "conversational",
            "description": "Friendly greeting"
        },
        {
            "query": "What is the latest news on AI regulation?",
            "expected_route": "research",
            "description": "Research question requiring web search"
        },
        {
            "query": "Thanks!",
            "expected_route": "conversational",
            "description": "Acknowledgment"
        },
    ]

    print("🧪 Testing Orchestrator Router Pattern\n")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['description']}")
        print(f"Query: '{test_case['query']}'")
        print(f"Expected route: {test_case['expected_route']}")

        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=test_case["query"])]
            }

            # Run only the router node to check decision
            result = graph.invoke(initial_state, {"recursion_limit": 50})

            # Check route decision
            route_decision = result.get("route_decision", "unknown")
            print(f"Actual route: {route_decision}")

            # Check if response was generated
            if len(result.get("messages", [])) > 1:
                response = result["messages"][-1].content
                print(f"Response preview: {response[:100]}...")

            # Validate
            if route_decision == test_case["expected_route"]:
                print("✅ PASS")
            else:
                print(f"❌ FAIL - Expected {test_case['expected_route']}, got {route_decision}")

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

        print("-" * 60)

    print("\n🎉 Router testing complete!")

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ Error: GEMINI_API_KEY not set in environment")
        print("Please set it in backend/.env file")
        sys.exit(1)

    test_router()
