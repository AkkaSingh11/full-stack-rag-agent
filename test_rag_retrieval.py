"""Test script to verify RAG retrieval is working."""

import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from dotenv import load_dotenv
from agent.vector_store import get_retriever

# Load environment variables
load_dotenv(Path(__file__).parent / "backend" / ".env")

def test_retrieval():
    """Test that we can retrieve documents from the vector store."""
    print("Testing RAG retrieval...")
    print("=" * 50)

    # Get retriever
    retriever = get_retriever(k=3)

    if retriever is None:
        print("❌ ERROR: Retriever is None. Vector store may not be initialized.")
        return False

    print("✅ Retriever initialized successfully")

    # Test query
    test_query = "What is this document about?"
    print(f"\nTest query: '{test_query}'")
    print("-" * 50)

    try:
        docs = retriever.invoke(test_query)

        if not docs:
            print("❌ No documents retrieved")
            return False

        print(f"✅ Retrieved {len(docs)} documents\n")

        for i, doc in enumerate(docs, 1):
            print(f"Document {i}:")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Page: {doc.metadata.get('page', 'N/A')}")
            print(f"  Content preview: {doc.page_content[:200]}...")
            print()

        return True

    except Exception as e:
        print(f"❌ Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_retrieval()

    if success:
        print("=" * 50)
        print("✅ RAG system is working correctly!")
        print("Embeddings are stored and retrieval is functional.")
    else:
        print("=" * 50)
        print("❌ RAG system test failed")
        sys.exit(1)
