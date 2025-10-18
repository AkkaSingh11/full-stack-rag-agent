"""Detailed strategy comparison with side-by-side results."""

import sys
from pathlib import Path
from typing import List
from langchain_core.documents import Document

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.vector_store import (
    get_semantic_retriever,
    get_bm25_retriever,
    get_hybrid_retriever
)


def format_doc_preview(doc: Document, max_length: int = 150) -> str:
    """Format document preview with metadata."""
    content = doc.page_content.replace('\n', ' ')[:max_length]
    source = doc.metadata.get('source_file', 'Unknown')
    page = doc.metadata.get('page', 'N/A')
    return f"{content}... (Source: {source}, Page: {page})"


def compare_results_detailed(query: str, k: int = 3):
    """Detailed side-by-side comparison of retrieval strategies."""

    print(f"\n{'='*100}")
    print(f"QUERY: {query}")
    print(f"{'='*100}\n")

    # Get retrievers
    semantic = get_semantic_retriever(k=k)
    keyword = get_bm25_retriever(k=k)
    hybrid = get_hybrid_retriever(k=k, alpha=0.5)

    # Retrieve documents
    semantic_docs = semantic.invoke(query) if semantic else []
    keyword_docs = keyword.invoke(query) if keyword else []
    hybrid_docs = hybrid.invoke(query) if hybrid else []

    # Display results side by side
    print(f"{'SEMANTIC (🔍)':<35} | {'KEYWORD (📝)':<35} | {'HYBRID (⚡ α=0.5)':<35}")
    print("-" * 100)

    max_results = max(len(semantic_docs), len(keyword_docs), len(hybrid_docs))

    for i in range(max_results):
        # Semantic result
        if i < len(semantic_docs):
            sem_preview = semantic_docs[i].page_content[:30].replace('\n', ' ')
            sem_source = semantic_docs[i].metadata.get('source_file', 'Unknown')[:15]
            sem_str = f"[{i+1}] {sem_preview}... ({sem_source})"
        else:
            sem_str = "-"

        # Keyword result
        if i < len(keyword_docs):
            kw_preview = keyword_docs[i].page_content[:30].replace('\n', ' ')
            kw_source = keyword_docs[i].metadata.get('source_file', 'Unknown')[:15]
            kw_str = f"[{i+1}] {kw_preview}... ({kw_source})"
        else:
            kw_str = "-"

        # Hybrid result
        if i < len(hybrid_docs):
            hyb_preview = hybrid_docs[i].page_content[:30].replace('\n', ' ')
            hyb_source = hybrid_docs[i].metadata.get('source_file', 'Unknown')[:15]
            hyb_str = f"[{i+1}] {hyb_preview}... ({hyb_source})"
        else:
            hyb_str = "-"

        print(f"{sem_str:<35} | {kw_str:<35} | {hyb_str:<35}")

    # Overlap analysis
    print(f"\n{'-'*100}")
    print("OVERLAP ANALYSIS:")

    semantic_content = {doc.page_content for doc in semantic_docs}
    keyword_content = {doc.page_content for doc in keyword_docs}
    hybrid_content = {doc.page_content for doc in hybrid_docs}

    sem_kw_overlap = len(semantic_content & keyword_content)
    sem_hyb_overlap = len(semantic_content & hybrid_content)
    kw_hyb_overlap = len(keyword_content & hybrid_content)

    print(f"  Semantic ∩ Keyword: {sem_kw_overlap}/{k} documents")
    print(f"  Semantic ∩ Hybrid:  {sem_hyb_overlap}/{k} documents")
    print(f"  Keyword ∩ Hybrid:   {kw_hyb_overlap}/{k} documents")

    # Unique contributions
    unique_to_semantic = len(semantic_content - keyword_content - hybrid_content)
    unique_to_keyword = len(keyword_content - semantic_content - hybrid_content)
    unique_to_hybrid = len(hybrid_content - semantic_content - keyword_content)

    print(f"\nUNIQUE CONTRIBUTIONS:")
    print(f"  Semantic only: {unique_to_semantic} documents")
    print(f"  Keyword only:  {unique_to_keyword} documents")
    print(f"  Hybrid only:   {unique_to_hybrid} documents")


def main():
    """Run detailed comparison tests."""

    print("\n" + "="*100)
    print("DETAILED STRATEGY COMPARISON - Side by Side Results")
    print("="*100)

    test_queries = [
        # Keyword-heavy (BM25 should find better matches)
        ("What is force majeure?", "keyword-focused"),

        # Semantic (embeddings should understand better)
        ("How do I handle contract violations?", "semantic-focused"),

        # Mixed (hybrid should excel)
        ("What legal requirements apply to agreements?", "mixed"),
    ]

    for query, query_type in test_queries:
        print(f"\n[Query Type: {query_type.upper()}]")
        compare_results_detailed(query)

    print(f"\n{'='*100}")
    print("KEY OBSERVATIONS:")
    print("="*100)
    print("""
1. OVERLAP: Low overlap indicates strategies retrieve different content
   - High overlap = strategies agree on relevance
   - Low overlap = complementary approaches (hybrid captures both)

2. UNIQUE CONTRIBUTIONS: Shows what each strategy uniquely finds
   - Semantic finds conceptually similar docs
   - Keyword finds exact term matches
   - Hybrid may find docs neither pure strategy would rank highly

3. RESULT ORDERING: Position matters (top result is most important)
   - Hybrid's ranking combines both relevance signals
   - Top result often differs across strategies
    """)


if __name__ == "__main__":
    main()
