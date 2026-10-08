"""Classic retrieval: similarity search over ChromaDB embeddings."""

from rag.index.store import get_vector_store


def get_retriever(k: int = 4):
    store = get_vector_store()
    return store.as_retriever(search_kwargs={"k": k})
