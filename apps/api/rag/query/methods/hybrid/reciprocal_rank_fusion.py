"""Reciprocal Rank Fusion (RRF) – combine BM25 and dense retrieval results.

RRF is a lightweight, parameter-free method for fusing ranked lists.
For each document, score = Σ 1 / (k + rank) across all retrievers,
where k is a constant (typically 60).

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.documents import Document

from rag.query.methods.hybrid.bm25 import bm25_retrieve
from rag.query.methods.hybrid.dense import dense_retrieve

DEFAULT_K = 60  # RRF constant — commonly set to 60


def _rrf_score(rank: int, k: int = DEFAULT_K) -> float:
    return 1.0 / (k + rank)


def reciprocal_rank_fusion(
    query: str,
    k: int = 10,
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    rrf_constant: int = DEFAULT_K,
) -> list[Document]:
    """Fuse BM25 and dense retrieval lists with Reciprocal Rank Fusion.

    Args:
        query: User question.
        k: Number of documents to return after fusion.
        bm25_top_k: Number of candidates to fetch from BM25.
        dense_top_k: Number of candidates to fetch from dense retrieval.
        rrf_constant: The k parameter in the RRF formula (default 60).

    Returns:
        Documents re-ranked by fused RRF score, highest first.
    """
    bm25_docs = bm25_retrieve(query, k=bm25_top_k)
    dense_docs = dense_retrieve(query, k=dense_top_k)

    # Assign RRF scores per list
    doc_scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        doc_scores[cid] = doc_scores.get(cid, 0.0) + _rrf_score(rank, rrf_constant)

    for rank, doc in enumerate(dense_docs, start=1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        doc_scores[cid] = doc_scores.get(cid, 0.0) + _rrf_score(rank, rrf_constant)

    # Sort by fused score descending
    fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[cid] for cid, _ in fused[:k]]


def reciprocal_rank_fusion_with_scores(
    query: str,
    k: int = 10,
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    rrf_constant: int = DEFAULT_K,
) -> list[tuple[Document, float]]:
    """Same as reciprocal_rank_fusion but returns (Document, RRF_score) pairs."""
    bm25_docs = bm25_retrieve(query, k=bm25_top_k)
    dense_docs = dense_retrieve(query, k=dense_top_k)

    doc_scores: dict[int, float] = {}
    doc_map: dict[int, Document] = {}

    for rank, doc in enumerate(bm25_docs, start=1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        doc_scores[cid] = doc_scores.get(cid, 0.0) + _rrf_score(rank, rrf_constant)

    for rank, doc in enumerate(dense_docs, start=1):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        doc_scores[cid] = doc_scores.get(cid, 0.0) + _rrf_score(rank, rrf_constant)

    fused = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[cid], float(score)) for cid, score in fused[:k]]
