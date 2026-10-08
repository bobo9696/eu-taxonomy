"""LLM-based document reranking. Placed in indexation because it treats the indexed docs as candidates to score."""

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag.config import get_llm

RETRIEVAL_K = 4
FETCH_K = 8


class RelevanceScore(BaseModel):
    score: float = Field(description="Relevance score between 0.0 (irrelevant) and 1.0 (directly answers the question)")


RERANK_PROMPT = """You are a relevance judge for a RAG system.
Rate how relevant the following FAQ entry is to the user's question.
Score strictly between 0.0 (irrelevant) and 1.0 (directly answers the question).

Question: {question}

FAQ Entry:
{content}

Relevance score:"""


def rerank(question: str, docs: list[Document], top_k: int = RETRIEVAL_K) -> list[Document]:
    """Rerank documents using an LLM judge and return the top_k."""
    if len(docs) <= top_k:
        return docs

    llm = get_llm()
    scored: list[tuple[float, Document]] = []

    for doc in docs:
        content = doc.page_content[:1500]
        prompt = RERANK_PROMPT.format(question=question, content=content)
        verdict = llm.with_structured_output(RelevanceScore, strict=False).invoke(prompt)
        scored.append((verdict.score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
