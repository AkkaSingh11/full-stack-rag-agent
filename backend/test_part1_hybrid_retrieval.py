"""Test script for Part 1: Hybrid RAG (no re-ranker)."""

import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.vector_store import (
    get_semantic_retriever,
    get_bm25_retriever,
    get_hybrid_retriever
)


def test_strategy(strategy_name: str, retriever, query: str):
    """Test a single retrieval strategy."""
    print(f"\n{strategy_name}:")
    print("-" * 60)

    try:
        docs = retriever.invoke(query)
        print(f"Retrieved {len(docs)} documents")

        for i, doc in enumerate(docs[:3], 1):
            preview = doc.page_content[:100].replace('\n', ' ')
            source = doc.metadata.get('source_file', 'Unknown')
            print(f"  [{i}] {source}: {preview}...")

    except Exception as e:
        print(f"  Error: {e}")


def test_all_strategies(query: str):
    """Test all retrieval strategies with the same query."""
    print(f"\n{'='*70}")
    print(f"Query: {query}")
    print(f"{'='*70}")

    # Test semantic
    semantic_retriever = get_semantic_retriever(k=5)
    if semantic_retriever:
        test_strategy("1. SEMANTIC SEARCH (🔍 Embedding-based)", semantic_retriever, query)

    # Test keyword
    bm25_retriever = get_bm25_retriever(k=5)
    if bm25_retriever:
        test_strategy("2. KEYWORD SEARCH (📝 BM25)", bm25_retriever, query)

    # Test hybrid (alpha=0.5 - balanced)
    hybrid_retriever = get_hybrid_retriever(k=5, alpha=0.5)
    if hybrid_retriever:
        test_strategy("3. HYBRID SEARCH (⚡ alpha=0.5, balanced)", hybrid_retriever, query)

    # Test hybrid (alpha=0.2 - keyword-focused)
    hybrid_keyword_retriever = get_hybrid_retriever(k=5, alpha=0.2)
    if hybrid_keyword_retriever:
        test_strategy("4. HYBRID SEARCH (alpha=0.2, keyword-focused)", hybrid_keyword_retriever, query)

    # Test hybrid (alpha=0.8 - semantic-focused)
    hybrid_semantic_retriever = get_hybrid_retriever(k=5, alpha=0.8)
    if hybrid_semantic_retriever:
        test_strategy("5. HYBRID SEARCH (alpha=0.8, semantic-focused)", hybrid_semantic_retriever, query)


if __name__ == "__main__":
    # Test queries that benefit from different strategies
    test_queries = [
        "What is the definition of force majeure?",  # Legal term (keyword good)
        "How do I handle contract breaches?",        # Conceptual (semantic good)
        "Section 12.3 compliance requirements",      # Exact reference (keyword good)
        "What are my rights if the other party violates the agreement?",  # Semantic good
    ]

    for query in test_queries:
        test_all_strategies(query)

    print(f"\n{'='*70}")
    print("Testing complete!")
    print(f"{'='*70}\n")
