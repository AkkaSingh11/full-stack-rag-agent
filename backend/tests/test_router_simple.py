#!/usr/bin/env python3
"""Simple router test - just test greeting"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
load_dotenv()

from langchain_core.messages import HumanMessage
from agent.graph import graph

print("Testing router with greeting: 'Hi'")

initial_state = {
    "messages": [HumanMessage(content="Hi")]
}

result = graph.invoke(initial_state, {"recursion_limit": 10})

print(f"\nRoute decision: {result.get('route_decision', 'unknown')}")
print(f"\nMessages count: {len(result.get('messages', []))}")

if len(result.get("messages", [])) > 1:
    response = result["messages"][-1].content
    print(f"\nResponse: {response}")

print("\n✅ Test complete!")
