"""Comprehensive analysis of Hybrid RAG implementation.

This script compares semantic, keyword, and hybrid retrieval strategies
with detailed metrics and insights.
"""

import sys
import time
from pathlib import Path
from collections import defaultdict

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.vector_store import (
    get_semantic_retriever,
    get_bm25_retriever,
    get_hybrid_retriever
)


def measure_retrieval_time(retriever, query: str, runs: int = 3):
    """Measure average retrieval time."""
    times = []
    for _ in range(runs):
        start = time.time()
        retriever.invoke(query)
        elapsed = time.time() - start
        times.append(elapsed * 1000)  # Convert to ms
    return sum(times) / len(times)


def analyze_result_diversity(docs):
    """Analyze diversity of retrieved documents."""
    sources = [doc.metadata.get('source_file', 'Unknown') for doc in docs]
    unique_sources = len(set(sources))
    return unique_sources, sources


def compare_strategies(query: str, k: int = 5):
    """Compare all strategies for a single query with detailed metrics."""
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    results = {}

    # Test each strategy
    strategies = [
        ("Semantic", get_semantic_retriever(k=k)),
        ("Keyword", get_bm25_retriever(k=k)),
        ("Hybrid (α=0.3)", get_hybrid_retriever(k=k, alpha=0.3)),
        ("Hybrid (α=0.5)", get_hybrid_retriever(k=k, alpha=0.5)),
        ("Hybrid (α=0.7)", get_hybrid_retriever(k=k, alpha=0.7)),
    ]

    for strategy_name, retriever in strategies:
        if retriever is None:
            continue

        # Retrieve documents
        docs = retriever.invoke(query)

        # Measure performance
        avg_time = measure_retrieval_time(retriever, query)
        unique_sources, sources = analyze_result_diversity(docs)

        results[strategy_name] = {
            'docs': docs,
            'time_ms': avg_time,
            'count': len(docs),
            'unique_sources': unique_sources,
            'sources': sources
        }

        # Display results
        print(f"\n{strategy_name}:")
        print(f"  ⏱️  Avg Retrieval Time: {avg_time:.1f}ms")
        print(f"  📄 Documents Retrieved: {len(docs)}")
        print(f"  🎯 Unique Sources: {unique_sources}")
        print(f"  Top 3 Results:")
        for i, doc in enumerate(docs[:3], 1):
            preview = doc.page_content[:120].replace('\n', ' ')
            source = doc.metadata.get('source_file', 'Unknown')
            print(f"     [{i}] {preview}...")

    return results


def generate_insights(all_results):
    """Generate insights from all test results."""
    print(f"\n{'='*80}")
    print("INSIGHTS & ANALYSIS")
    print(f"{'='*80}")

    # Performance analysis
    print("\n📊 PERFORMANCE METRICS:")
    print("-" * 80)

    avg_times = defaultdict(list)
    for query_results in all_results.values():
        for strategy, data in query_results.items():
            avg_times[strategy].append(data['time_ms'])

    print(f"{'Strategy':<25} {'Avg Time (ms)':<15} {'Speedup vs Semantic':<20}")
    print("-" * 80)

    semantic_avg = sum(avg_times['Semantic']) / len(avg_times['Semantic'])
    for strategy in sorted(avg_times.keys()):
        avg = sum(avg_times[strategy]) / len(avg_times[strategy])
        speedup = semantic_avg / avg if avg > 0 else 0
        speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x slower"
        print(f"{strategy:<25} {avg:>10.1f}ms     {speedup_str:<20}")

    # Diversity analysis
    print("\n🎯 RESULT DIVERSITY:")
    print("-" * 80)

    diversity_scores = defaultdict(list)
    for query_results in all_results.values():
        for strategy, data in query_results.items():
            diversity = data['unique_sources'] / data['count'] if data['count'] > 0 else 0
            diversity_scores[strategy].append(diversity)

    print(f"{'Strategy':<25} {'Avg Diversity Score':<20}")
    print("-" * 80)
    for strategy in sorted(diversity_scores.keys()):
        avg_diversity = sum(diversity_scores[strategy]) / len(diversity_scores[strategy])
        print(f"{strategy:<25} {avg_diversity:>15.2%}")

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 80)

    keyword_avg = sum(avg_times['Keyword']) / len(avg_times['Keyword'])
    hybrid_05_avg = sum(avg_times['Hybrid (α=0.5)']) / len(avg_times['Hybrid (α=0.5)'])

    print(f"""
1. **For Speed-Critical Applications:**
   - Use Keyword (BM25): ~{keyword_avg:.0f}ms average
   - Best for: Exact term matching, product codes, specific sections

2. **For Semantic Understanding:**
   - Use Semantic: ~{semantic_avg:.0f}ms average
   - Best for: Conceptual queries, paraphrased questions

3. **For Best Overall Performance:**
   - Use Hybrid (α=0.5): ~{hybrid_05_avg:.0f}ms average
   - Balances speed and quality
   - Robust across query types

4. **For Fine-Tuning:**
   - α=0.3: More keyword-focused (good for technical docs)
   - α=0.5: Balanced (recommended default)
   - α=0.7: More semantic-focused (good for conceptual queries)
    """)


def main():
    """Run comprehensive hybrid RAG analysis."""

    # Test queries covering different scenarios
    test_queries = [
        # Keyword-heavy queries (BM25 should excel)
        ("What is the definition of force majeure?", "keyword-heavy"),
        ("Section 12.3 compliance requirements", "exact-reference"),

        # Semantic queries (embedding should excel)
        ("How do I handle contract breaches?", "conceptual"),
        ("What are my rights if the other party violates the agreement?", "semantic"),

        # Mixed queries (hybrid should excel)
        ("What legal requirements apply to property transfer agreements?", "mixed"),
        ("How to draft a non-disclosure agreement for business partnerships?", "mixed"),
    ]

    all_results = {}

    print("\n" + "="*80)
    print("HYBRID RAG IMPLEMENTATION ANALYSIS")
    print("="*80)
    print(f"\nTesting {len(test_queries)} queries across 5 retrieval strategies...")

    for query, query_type in test_queries:
        print(f"\n[Query Type: {query_type}]")
        results = compare_strategies(query)
        all_results[query] = results

    # Generate comprehensive insights
    generate_insights(all_results)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
