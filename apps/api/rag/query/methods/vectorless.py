"""Vectorless retrieval: the LLM navigates the FAQ tree (section -> question) instead of searching embeddings."""

from collections import defaultdict
from functools import lru_cache

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag.config import get_llm
from rag.index.methods.basic import load_documents

MAX_SECTIONS = 3

SECTION_PROMPT = """You are navigating the table of contents of the EU Taxonomy FAQ to find where a question is answered.

Sections:
{sections}

Question: {question}

Select the sections most likely to contain the answer (at most {max_sections}), most relevant first."""

ENTRY_PROMPT = """You are navigating the EU Taxonomy FAQ to find the entries that answer a question.

FAQ entries:
{entries}

Question: {question}

Select the entries whose answer most likely answers the question (at most {k}), most relevant first. Select none if no entry is relevant."""


class SectionSelection(BaseModel):
    section_ids: list[int] = Field(description="IDs of the selected sections, most relevant first")


class EntrySelection(BaseModel):
    entry_ids: list[int] = Field(description="IDs of the selected FAQ entries, most relevant first")


@lru_cache
def _sections() -> list[tuple[str, list[Document]]]:
    """FAQ entries grouped by section, in document order."""
    tree: dict[str, list[Document]] = defaultdict(list)
    for doc in load_documents():
        tree[doc.metadata["section"]].append(doc)
    return list(tree.items())


def vectorless_retrieve(question: str, k: int) -> list[Document]:
    sections = _sections()
    llm = get_llm()

    section_ids = llm.with_structured_output(SectionSelection).invoke(
        SECTION_PROMPT.format(
            sections="\n".join(f"[{i}] {title} ({len(docs)} questions)" for i, (title, docs) in enumerate(sections)),
            question=question,
            max_sections=MAX_SECTIONS,
        )
    ).section_ids
    candidates = {
        doc.metadata["chunk_id"]: doc
        for i in list(dict.fromkeys(section_ids))[:MAX_SECTIONS]
        if 0 <= i < len(sections)
        for doc in sections[i][1]
    }
    if not candidates:
        return []

    entry_ids = llm.with_structured_output(EntrySelection).invoke(
        ENTRY_PROMPT.format(
            entries="\n".join(f"[{chunk_id}] {doc.metadata['question']}" for chunk_id, doc in candidates.items()),
            question=question,
            k=k,
        )
    ).entry_ids
    return [candidates[i] for i in dict.fromkeys(entry_ids) if i in candidates][:k]
