"""Parse EU Taxonomy FAQ markdown into structured Q&A chunks."""

import re
from pathlib import Path
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

FAQ_PATH = Path(__file__).parent.parent.parent.parent / "data" / "taxonomy_faqs_cleaned.md"

from typing import Optional


@dataclass
class FAQChunk:
    section: str
    question: str
    answer: str
    id: int
    sectors: Optional[list[str]] = None


def parse_taxonomy_faqs(md_path: str | Path) -> List[FAQChunk]:
    """
    Parse the markdown file into FAQ chunks.
    Structure:
      ## Section Title
      ### Question text?
      Answer paragraphs...
      ### Next question...
    """
    text = Path(md_path).read_text(encoding="utf-8")
    chunks: List[FAQChunk] = []
    current_section = "General"
    current_question = None
    current_answer_lines: List[str] = []
    chunk_id = 0

    lines = text.splitlines()
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()

        if line.startswith("## ") and not line.startswith("### "):
            if current_question and current_answer_lines:
                chunks.append(
                    FAQChunk(
                        section=current_section,
                        question=clean_markdown(current_question),
                        answer=clean_markdown("\n".join(current_answer_lines)),
                        id=chunk_id,
                    )
                )
                chunk_id += 1
                current_question = None
                current_answer_lines = []
            current_section = line[3:].strip()

        elif line.startswith("### "):
            if current_question and current_answer_lines:
                chunks.append(
                    FAQChunk(
                        section=current_section,
                        question=clean_markdown(current_question),
                        answer=clean_markdown("\n".join(current_answer_lines)),
                        id=chunk_id,
                    )
                )
                chunk_id += 1
                current_answer_lines = []
            current_question = line[3:].strip()

        elif current_question is not None:
            current_answer_lines.append(lines[idx])

        idx += 1

    if current_question and current_answer_lines:
        chunks.append(
            FAQChunk(
                section=current_section,
                question=clean_markdown(current_question),
                answer=clean_markdown("\n".join(current_answer_lines)),
                id=chunk_id,
            )
        )

    return chunks


def clean_markdown(text: str) -> str:
    """Remove markdown images/artifacts and normalize whitespace."""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\(\d+\)(See [^)]+)?", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

@lru_cache
def load_documents() -> List[Document]:
    """FAQ entries as LangChain documents, shared by every retriever."""
    return [
        Document(
            page_content=f"Section: {chunk.section}\nQuestion: {chunk.question}\nAnswer: {chunk.answer}",
            metadata={
                "section": chunk.section,
                "question": chunk.question,
                "chunk_id": chunk.id,
            },
        )
        for chunk in parse_taxonomy_faqs(FAQ_PATH)
    ]
