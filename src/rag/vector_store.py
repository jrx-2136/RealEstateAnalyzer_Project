# src/rag/vector_store.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTOR_DIR = "data/vectorstore"


def build_or_load_vector_store(text_documents: list[str]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(VECTOR_DIR):
        print("🔁 Loading existing vector store (local embeddings)")
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)



    print("🧠 Building vector store (local embeddings, one-time)")
    vector_db = FAISS.from_texts(text_documents, embeddings)
    vector_db.save_local(VECTOR_DIR)
    return vector_db
