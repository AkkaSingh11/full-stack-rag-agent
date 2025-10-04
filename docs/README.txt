RAG Document Directory
======================

This directory is used to store documents for the RAG (Retrieval-Augmented Generation) system.

Supported file formats:
- PDF (.pdf)
- DOCX (.docx)

How to ingest documents:
1. Place your PDF or DOCX files in this directory
2. Run: python backend/scripts/ingest_documents.py
3. The documents will be chunked and embedded into the ChromaDB vector store
4. You can then query the documents through the agent

Example queries after ingestion:
- "What does the document say about [topic]?"
- "According to the uploaded files, what is [question]?"
- "Summarize the main points from the documents"

The agent will automatically route these queries to the RAG path and retrieve relevant information from your documents.

If the documents don't contain sufficient information, the agent will automatically fall back to web research.
