"""Dense retrieval over EU Taxonomy FAQ documents using ChromaDB embeddings.

Dense retrieval encodes both queries and documents into the same embedding
space and returns the nearest neighbours by cosine similarity.
"""

from __future__ import annotations

from langchain_core.documents import Document

from rag.index.store import get_vector_store


def dense_retrieve(query: str, k: int = 10) -> list[Document]:
    """Retrieve the top-k documents using dense vector similarity.

    Args:
        query: User question.
        k: Number of documents to return.

    Returns:
        List of Documents ranked by embedding similarity descending.
    """
    store = get_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def dense_retrieve_with_scores(query: str, k: int = 10) -> list[tuple[Document, float]]:
    """Same as dense_retrieve but returns (Document, similarity_score) pairs.

    Chroma returns distance by default; we convert it to a similarity score.
    """
    store = get_vector_store()
    # similarity_search_with_relevance_scores returns (doc, score) where higher is better
    results = store.similarity_search_with_relevance_scores(query, k=k)
    return results
