"""RAG pipeline: pluggable retrieval (classic or vectorless) feeding a shared generation step."""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from rag.config import get_llm
from rag.index.rerank import FETCH_K, RETRIEVAL_K, rerank
from rag.query.methods.classic import get_retriever
from rag.query.methods.vectorless import vectorless_retrieve
from rag.query.methods.sector_aware import sector_aware_retrieve
from rag.query.methods.hybrid.cross_encoder_rerank import cross_encoder_rerank
from rag.prompts import PROMPTS

RagType = Literal["classic", "vectorless", "sector_aware", "hybrid"]
PromptType = Literal["basic", "intermediate", "expert"]
TOP_K = RETRIEVAL_K


def retrieve(question: str, rag_type: RagType, use_rerank: bool = False) -> list[Document]:
    k = FETCH_K if use_rerank else RETRIEVAL_K
    if rag_type == "vectorless":
        docs = vectorless_retrieve(question, k=k)
    elif rag_type == "sector_aware":
        docs = sector_aware_retrieve(question, k=k)
    elif rag_type == "hybrid":
        docs = cross_encoder_rerank(question, top_k=RETRIEVAL_K)
    else:
        docs = get_retriever(k=k).invoke(question)

    if use_rerank and len(docs) > RETRIEVAL_K:
        docs = rerank(question, docs, top_k=RETRIEVAL_K)
    return docs


def ask_question(
    question: str,
    prompt_type: PromptType = "intermediate",
    rag_type: RagType = "classic",
    use_rerank: bool = False,
):
    import time
    t0 = time.perf_counter()
    docs = retrieve(question, rag_type, use_rerank=use_rerank)
    t1 = time.perf_counter()
    chain = ChatPromptTemplate.from_template(PROMPTS[prompt_type]) | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": "\n\n---\n\n".join(d.page_content for d in docs), "question": question})
    t2 = time.perf_counter()
    return {
        "answer": answer,
        "contexts": [d.page_content for d in docs],
        "sources": [
            {
                "section": d.metadata.get("section"),
                "question": d.metadata.get("question"),
                "chunk_id": d.metadata.get("chunk_id"),
            }
            for d in docs
        ],
        "retrieval_ms": round((t1 - t0) * 1000, 1),
        "generation_ms": round((t2 - t1) * 1000, 1),
        "total_ms": round((t2 - t0) * 1000, 1),
    }
