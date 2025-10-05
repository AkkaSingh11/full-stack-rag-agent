"""Vector store management for RAG using ChromaDB and Gemini embeddings."""

import os
from pathlib import Path
from typing import List, Optional, Literal

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize

# Download punkt tokenizer if not available
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Get the project root directory (3 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Configuration
SOURCE_DIR = PROJECT_ROOT / "docs"
INDEX_DIR = PROJECT_ROOT / "backend" / "chroma_db"
COLLECTION = "kb_collection"
EMBED_MODEL = "gemini-embedding-001"

# Global vector store instance (singleton pattern)
_vector_store = None

# Global BM25 index (singleton pattern)
_bm25_index = None
_bm25_documents = None
_bm25_metadatas = None
_bm25_ids = None


def _tokenize(text: str) -> List[str]:
    """Simple tokenizer for BM25."""
    return word_tokenize(text.lower())


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


def _build_bm25_index():
    """Build BM25 index from Chroma documents (singleton pattern)."""
    global _bm25_index, _bm25_documents, _bm25_metadatas, _bm25_ids

    if _bm25_index is not None:
        return _bm25_index

    # Get all documents from vector store
    vectordb = get_or_create_vector_store()
    if vectordb is None:
        return None

    # Retrieve all documents from Chroma
    try:
        all_data = vectordb.get()
        _bm25_documents = all_data['documents']
        _bm25_metadatas = all_data.get('metadatas', [])
        _bm25_ids = all_data.get('ids', [])

        if not _bm25_documents:
            return None

        # Tokenize all documents
        tokenized_docs = [_tokenize(doc) for doc in _bm25_documents]

        # Build BM25 index
        _bm25_index = BM25Okapi(tokenized_docs)

        print(f"Built BM25 index with {len(_bm25_documents)} documents")
        return _bm25_index
    except Exception as e:
        print(f"Error building BM25 index: {e}")
        return None


def get_semantic_retriever(k: int = 3) -> BaseRetriever:
    """Get dense/semantic retriever (current Chroma retriever)."""
    vectordb = get_or_create_vector_store()
    if vectordb:
        return vectordb.as_retriever(search_kwargs={"k": k})
    return None


def get_bm25_retriever(k: int = 3) -> BaseRetriever:
    """Get sparse/keyword retriever using BM25."""

    class BM25Retriever(BaseRetriever):
        k: int = 3

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            bm25_index = _build_bm25_index()

            if bm25_index is None or _bm25_documents is None:
                return []

            # Tokenize query
            tokenized_query = _tokenize(query)

            # Get BM25 scores
            scores = bm25_index.get_scores(tokenized_query)

            # Get top-k indices
            top_k_indices = scores.argsort()[-self.k:][::-1]

            # Build document objects
            docs = []
            for idx in top_k_indices:
                if idx < len(_bm25_documents):
                    doc = Document(
                        page_content=_bm25_documents[idx],
                        metadata=_bm25_metadatas[idx] if _bm25_metadatas and idx < len(_bm25_metadatas) else {}
                    )
                    docs.append(doc)

            return docs

    return BM25Retriever(k=k)


def get_hybrid_retriever(k: int = 3, alpha: float = 0.5) -> BaseRetriever:
    """Get hybrid retriever combining semantic + keyword search.

    Args:
        k: Number of documents to retrieve
        alpha: Balance between sparse (0) and dense (1).
               0.0 = pure BM25, 0.5 = balanced, 1.0 = pure semantic

    Returns:
        EnsembleRetriever that combines both approaches
    """
    semantic_retriever = get_semantic_retriever(k=k)
    bm25_retriever = get_bm25_retriever(k=k)

    if semantic_retriever and bm25_retriever:
        return EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever],
            weights=[1 - alpha, alpha]  # BM25 weight, Semantic weight
        )
    return semantic_retriever  # Fallback to semantic if BM25 unavailable


def get_retriever_by_strategy(
    strategy: Literal["semantic", "keyword", "hybrid"],
    k: int = 3,
    alpha: float = 0.5
) -> BaseRetriever:
    """Get retriever based on strategy configuration.

    Args:
        strategy: One of "semantic", "keyword", or "hybrid"
        k: Number of documents to retrieve
        alpha: Hybrid balance (only used for "hybrid" strategy)

    Returns:
        Configured retriever instance
    """
    if strategy == "semantic":
        return get_semantic_retriever(k=k)
    elif strategy == "keyword":
        return get_bm25_retriever(k=k)
    elif strategy == "hybrid":
        return get_hybrid_retriever(k=k, alpha=alpha)
    else:
        return get_semantic_retriever(k=k)  # Default fallback
