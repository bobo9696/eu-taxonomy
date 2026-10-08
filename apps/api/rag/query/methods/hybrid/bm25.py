"""BM25 sparse retrieval over EU Taxonomy FAQ documents.

BM25 (Best Match 25) is a probabilistic ranking function that scores
documents based on term frequency, inverse document frequency, and
document length normalization — without needing embeddings.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag.index.methods.basic import load_documents


@lru_cache
def _get_bm25_index() -> tuple[BM25Okapi, list[Document]]:
    """Build and cache the BM25 index over all FAQ documents."""
    docs = load_documents()
    # Simple whitespace tokenization; BM25 expects list[list[str]]
    tokenized_corpus = [doc.page_content.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, docs


def bm25_retrieve(query: str, k: int = 10) -> list[Document]:
    """Retrieve the top-k documents using BM25 sparse scoring.

    Args:
        query: User question.
        k: Number of documents to return.

    Returns:
        List of Documents ranked by BM25 score descending.
    """
    bm25, docs = _get_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair (score, doc), sort descending by score, take top-k
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def bm25_retrieve_with_scores(query: str, k: int = 10) -> list[tuple[Document, float]]:
    """Same as bm25_retrieve but returns (Document, BM25_score) pairs."""
    bm25, docs = _get_bm25_index()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [(doc, float(score)) for score, doc in scored[:k]]
