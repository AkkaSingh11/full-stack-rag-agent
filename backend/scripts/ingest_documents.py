"""CLI tool for ingesting documents into the RAG knowledge base.

Usage:
    python backend/scripts/ingest_documents.py

This will ingest all PDF and DOCX files from the 'docs' directory.
"""

import os
import sys
from pathlib import Path

# Add backend/src to path so we can import agent modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

from agent.vector_store import create_vector_store

# Load environment variables
load_dotenv()


def main():
    """Ingest all documents from the docs directory."""
    if os.getenv("GEMINI_API_KEY") is None:
        print("Error: GEMINI_API_KEY is not set in environment")
        print("Please set GEMINI_API_KEY in your .env file")
        sys.exit(1)

    print("Starting document ingestion...")
    print("=" * 50)

    vector_store = create_vector_store()

    if vector_store is None:
        print("\nIngestion failed. Please check the error messages above.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Document ingestion completed successfully!")
    print("You can now query the knowledge base using the RAG agent.")


if __name__ == "__main__":
    main()
