"""Test script to verify ChromaDB collection details."""

import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

import os
from dotenv import load_dotenv
from agent.vector_store import get_or_create_vector_store

# Load environment variables
load_dotenv(Path(__file__).parent / "backend" / ".env")

def test_db_details():
    """Check ChromaDB collection details."""
    print("Checking ChromaDB collection details...")
    print("=" * 50)

    vectordb = get_or_create_vector_store()

    if vectordb is None:
        print("❌ ERROR: Vector store is None")
        return False

    try:
        # Get collection
        collection = vectordb._collection

        # Count documents
        count = collection.count()
        print(f"✅ Total documents in collection: {count}")

        # Get sample of data
        results = collection.peek(limit=5)

        print(f"\n📊 Collection Statistics:")
        print(f"  - Collection name: {collection.name}")
        print(f"  - Total chunks: {count}")

        if results and 'metadatas' in results:
            print(f"\n📄 Sample Document Metadata:")
            for i, metadata in enumerate(results['metadatas'][:3], 1):
                print(f"  Document {i}:")
                for key, value in metadata.items():
                    print(f"    {key}: {value}")
                print()

        # Check embeddings (handle numpy array properly)
        if results and 'embeddings' in results and results['embeddings'] is not None:
            if len(results['embeddings']) > 0:
                embedding_dim = len(results['embeddings'][0])
                print(f"\n📐 Embedding Dimensions: {embedding_dim}")
                print(f"📊 Number of embeddings in sample: {len(results['embeddings'])}")

        print("\n✅ ChromaDB is properly configured with embeddings!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_db_details()
    sys.exit(0 if success else 1)
