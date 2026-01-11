# src/rag/vector_store.py

"""
Vector store for property embeddings using FAISS.
Provides semantic search over property documents.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTOR_DIR = "data/vectorstore"


def get_embeddings():
    """Get the embedding model instance."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def build_or_load_vector_store(text_documents: list[str], force_rebuild: bool = False):
    """
    Build a new vector store or load existing one.
    
    Args:
        text_documents: List of text documents to embed
        force_rebuild: If True, rebuild even if exists
        
    Returns:
        FAISS vector store instance
    """
    embeddings = get_embeddings()

    if os.path.exists(VECTOR_DIR) and not force_rebuild:
        print("🔁 Loading existing vector store (local embeddings)")
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

    print(f"🧠 Building vector store with {len(text_documents)} documents...")
    vector_db = FAISS.from_texts(text_documents, embeddings)
    vector_db.save_local(VECTOR_DIR)
    print(f"✅ Vector store saved to {VECTOR_DIR}")
    return vector_db


def similarity_search_with_score(vector_db, query: str, k: int = 5, score_threshold: float = None):
    """
    Perform similarity search with relevance scores.
    
    Args:
        vector_db: FAISS vector store
        query: Search query
        k: Number of results
        score_threshold: Minimum score to include (lower is better for FAISS L2 distance)
        
    Returns:
        List of (document, score) tuples
    """
    results = vector_db.similarity_search_with_score(query, k=k)
    
    if score_threshold is not None:
        # FAISS returns L2 distance - lower is better
        # Filter out low-relevance results
        results = [(doc, score) for doc, score in results if score < score_threshold]
    
    return results


def rebuild_vector_store(text_documents: list[str]):
    """
    Force rebuild the vector store with new documents.
    Useful when CSV data is updated.
    """
    # Delete existing if present
    if os.path.exists(VECTOR_DIR):
        import shutil
        shutil.rmtree(VECTOR_DIR)
        print(f"🗑️ Removed old vector store")
    
    return build_or_load_vector_store(text_documents, force_rebuild=True)
