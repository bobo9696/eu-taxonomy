"""Hybrid retrieval combining BM25 sparse + dense vector + RRF fusion + cross-encoder reranking."""

from rag.query.methods.hybrid.bm25 import bm25_retrieve, bm25_retrieve_with_scores
from rag.query.methods.hybrid.dense import dense_retrieve, dense_retrieve_with_scores
from rag.query.methods.hybrid.reciprocal_rank_fusion import (
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_with_scores,
)
from rag.query.methods.hybrid.cross_encoder_rerank import (
    cross_encoder_rerank,
    cross_encoder_rerank_with_scores,
)

__all__ = [
    "bm25_retrieve",
    "bm25_retrieve_with_scores",
    "dense_retrieve",
    "dense_retrieve_with_scores",
    "reciprocal_rank_fusion",
    "reciprocal_rank_fusion_with_scores",
    "cross_encoder_rerank",
    "cross_encoder_rerank_with_scores",
]
