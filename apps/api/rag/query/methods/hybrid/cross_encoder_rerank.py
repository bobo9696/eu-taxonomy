"""Cross-Encoder Reranking – rerank retrieved documents with a cross-encoder model.

A bi-encoder (dense retrieval) embeds query and docs independently, which is fast
but can miss fine-grained interactions. A cross-encoder feeds [query, doc] pairs
through a transformer jointly, yielding a much more accurate relevance score —
but is slower because it cannot be pre-computed.

Workflow:
  1. Retrieve a large candidate pool (e.g. 50–100 docs) with BM25, dense, or RRF.
  2. Score every (query, doc) pair with the cross-encoder.
  3. Return the top-k (e.g. top 10) by the cross-encoder score.
"""

from __future__ import annotations

import os
from typing import Sequence

from langchain_core.documents import Document

from rag.query.methods.hybrid.bm25 import bm25_retrieve
from rag.query.methods.hybrid.dense import dense_retrieve
from rag.query.methods.hybrid.reciprocal_rank_fusion import reciprocal_rank_fusion

_DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Pre-load the model eagerly in the main process to avoid meta-tensor issues
# when the model is first accessed from an asyncio thread.
_tokenizer = None
_model = None


def _ensure_loaded():
    """Lazy-load but safe to call from any thread because the heavy lifting is already done."""
    global _tokenizer, _model
    if _tokenizer is not None:
        return _tokenizer, _model

    import torch
    from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_DEFAULT_CROSS_ENCODER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        _DEFAULT_CROSS_ENCODER_MODEL,
        low_cpu_mem_usage=False,   # <-- prevents meta tensors
        device_map=None,             # <-- no accelerate offloading
    )
    model.eval()

    _tokenizer = tokenizer
    _model = model
    return _tokenizer, _model


def _cross_encode_score(query: str, texts: list[str]) -> list[float]:
    """Score a batch of (query, text) pairs. Returns list of float scores."""
    import torch

    tokenizer, model = _ensure_loaded()
    features = tokenizer(
        [query] * len(texts),
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**features)
        scores = outputs.logits.squeeze(-1)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
        scores = torch.sigmoid(scores)

    return scores.tolist()


def cross_encoder_rerank(
    query: str,
    candidate_pool: list[Document] | None = None,
    top_k: int = 10,
    pool_k: int = 50,
) -> list[Document]:
    """Rerank candidates with a cross-encoder and return the top-k."""
    if candidate_pool is None:
        candidate_pool = reciprocal_rank_fusion(query, k=pool_k)

    if not candidate_pool:
        return []

    texts = [doc.page_content for doc in candidate_pool]
    scores = _cross_encode_score(query, texts)

    scored = sorted(zip(scores, candidate_pool), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


def cross_encoder_rerank_with_scores(
    query: str,
    candidate_pool: list[Document] | None = None,
    top_k: int = 10,
    pool_k: int = 50,
) -> list[tuple[Document, float]]:
    """Same as cross_encoder_rerank but returns (Document, cross_encoder_score) pairs."""
    if candidate_pool is None:
        candidate_pool = reciprocal_rank_fusion(query, k=pool_k)

    if not candidate_pool:
        return []

    texts = [doc.page_content for doc in candidate_pool]
    scores = _cross_encode_score(query, texts)

    scored = sorted(zip(scores, candidate_pool), key=lambda x: x[0], reverse=True)
    return [(doc, float(score)) for score, doc in scored[:top_k]]
