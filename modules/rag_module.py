from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os

_vectorstore = None


def _load_vectorstore():
    """Load or build FAISS vector store from local captions database."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    db_path = "vector_store/captions_db.txt"
    if not os.path.exists(db_path):
        return None

    with open(db_path, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f.readlines() if line.strip()]

    documents = [Document(page_content=c) for c in captions]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    _vectorstore = FAISS.from_documents(documents, embeddings)
    return _vectorstore


def retrieve_similar_captions(caption: str, k: int = 3) -> list:
    """Retrieve k most similar captions from local knowledge base."""
    try:
        vectorstore = _load_vectorstore()
        if vectorstore is None:
            return []

        results = vectorstore.similarity_search(caption, k=k)
        return [doc.page_content for doc in results]

    except Exception as e:
        print(f"RAG Error: {e}")
        return []