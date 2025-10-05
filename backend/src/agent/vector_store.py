"""Vector store management for RAG using ChromaDB and Gemini embeddings."""

import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Configuration
SOURCE_DIR = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "backend" / "chroma_db"
COLLECTION = "kb_collection"
EMBED_MODEL = "gemini-embedding-001"

# Global vector store instance (singleton pattern)
_vector_store = None


def load_documents(folder_path: str) -> List[Document]:
    """Load documents from a folder path.

    Supports PDF and DOCX file formats.

    Args:
        folder_path: Path to the folder containing documents

    Returns:
        List of loaded documents
    """
    documents = []
    if not os.path.exists(folder_path):
        return documents

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue

        documents.extend(loader.load())

    return documents


def create_vector_store() -> Optional[Chroma]:
    """Create a new vector store from documents in SOURCE_DIR.

    Returns:
        Initialized Chroma vector store or None if no documents found
    """
    if not os.path.exists(SOURCE_DIR):
        os.makedirs(SOURCE_DIR)
        print(f"Created '{SOURCE_DIR}' directory. Please add your documents to it.")
        return None

    documents = load_documents(str(SOURCE_DIR))
    if not documents:
        print("No documents found to index.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR),
        collection_name=COLLECTION,
    )
    vectordb.persist()
    print(f"Index built at {INDEX_DIR.resolve()} with {len(chunks)} chunks")
    return vectordb


def get_or_create_vector_store() -> Optional[Chroma]:
    """Get or create the vector store instance (singleton pattern).

    Returns:
        Chroma vector store instance or None if unavailable
    """
    global _vector_store

    if _vector_store is not None:
        return _vector_store

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL, google_api_key=os.getenv("GEMINI_API_KEY")
    )

    if INDEX_DIR.exists():
        # Load existing vector store
        _vector_store = Chroma(
            persist_directory=str(INDEX_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION,
        )
    else:
        # Create new vector store
        _vector_store = create_vector_store()

    return _vector_store


def ingest_single_document(file_path: str, filename: str) -> dict:
    """Ingest a single document into the existing vector store.

    Args:
        file_path: Path to the document file
        filename: Original filename for metadata

    Returns:
        dict with status, message, and chunks count
    """
    try:
        # Load document based on file type
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            return {"status": "error", "message": f"Unsupported file type: {filename}"}

        documents = loader.load()

        if not documents:
            return {"status": "error", "message": "No content extracted from document"}

        # Add filename to metadata for tracking
        for doc in documents:
            doc.metadata["source_file"] = filename

        # Chunk the documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Get or create vector store
        vectordb = get_or_create_vector_store()

        if vectordb is None:
            return {"status": "error", "message": "Failed to initialize vector store"}

        # Add chunks to existing vector store
        vectordb.add_documents(chunks)
        vectordb.persist()

        return {
            "status": "success",
            "message": f"Successfully ingested {filename}",
            "chunks": len(chunks),
        }

    except Exception as e:
        return {"status": "error", "message": f"Error ingesting {filename}: {str(e)}"}


def get_retriever(k: int = 3):
    """Get a retriever from the vector store.

    Args:
        k: Number of documents to retrieve

    Returns:
        Retriever instance or None if vector store unavailable
    """
    vectordb = get_or_create_vector_store()
    if vectordb:
        return vectordb.as_retriever(search_kwargs={"k": k})
    return None
