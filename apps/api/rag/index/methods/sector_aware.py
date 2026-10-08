"""Sector-aware chunking: each FAQ chunk is tagged with its economic sector.

The EU Taxonomy Climate Delegated Act organises activities by NACE-related
sections (1.1–10.2).  We map every chunk to the sector(s) it mentions so that
the retriever can later filter by sector before doing similarity search.
"""

import re
from pathlib import Path
from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from .basic import FAQ_PATH, FAQChunk, parse_taxonomy_faqs, clean_markdown

# ---------------------------------------------------------------------------
# Section → sector mapping (EU Taxonomy Climate Delegated Act)
# ---------------------------------------------------------------------------
SECTION_TO_SECTOR: dict[str, str] = {
    # Forestry
    "1.1": "Forestry",
    "1.2": "Forestry",
    "1.3": "Forestry",
    "1.4": "Forestry",
    # Manufacturing
    "3.1": "Manufacturing",
    "3.2": "Manufacturing",
    "3.3": "Manufacturing",
    "3.5": "Manufacturing",
    "3.6": "Manufacturing",
    "3.7": "Manufacturing",
    "3.8": "Manufacturing",
    "3.9": "Manufacturing",
    "3.10": "Manufacturing",
    "3.17": "Manufacturing",
    # Energy
    "4.3": "Energy",
    "4.5": "Energy",
    "4.6": "Energy",
    "4.7": "Energy",
    "4.8": "Energy",
    "4.9": "Energy",
    "4.10": "Energy",
    "4.12": "Energy",
    "4.13": "Energy",
    "4.14": "Energy",
    "4.16": "Energy",
    # Water, waste & circular economy
    "5.1": "Water & Waste",
    "5.2": "Water & Waste",
    "5.3": "Water & Waste",
    "5.9": "Water & Waste",
    "5.11": "Water & Waste",
    # Transport
    "6.1": "Transport",
    "6.2": "Transport",
    "6.4": "Transport",
    "6.5": "Transport",
    "6.6": "Transport",
    "6.8": "Transport",
    "6.10": "Transport",
    "6.11": "Transport",
    "6.14": "Transport",
    "6.15": "Transport",
    # Construction & buildings
    "7.1": "Construction",
    "7.2": "Construction",
    "7.3": "Construction",
    "7.6": "Construction",
    "7.7": "Construction",
    # Information & communication
    "8.1": "ICT",
    "8.2": "ICT",
    "8.3": "ICT",
    # Professional services
    "9.1": "Professional Services",
    "9.2": "Professional Services",
    "9.3": "Professional Services",
    # Financial / insurance
    "10.1": "Financial & Insurance",
    "10.2": "Financial & Insurance",
}

_KNOWN_SECTIONS = set(SECTION_TO_SECTOR.keys())


def _extract_sectors(text: str) -> set[str]:
    """Return the set of economic sectors mentioned in the text."""
    matches = re.findall(r"Section[s]?\s+([0-9]+\.[0-9]+)", text)
    sectors: set[str] = set()
    for sec in matches:
        if sec in _KNOWN_SECTIONS:
            sectors.add(SECTION_TO_SECTOR[sec])
    return sectors


def _classify_by_section(section_name: str) -> str | None:
    """Map the FAQ section title (e.g. 'Climate Delegated Act') to a coarse sector."""
    # Only the Climate Delegated Act has fine-grained activity sections.
    # The other top-level sections are already sector-ish.
    MAPPING = {
        "Climate Delegated Act": None,          # → will be resolved per-chunk
        "Complementary Climate Delegated Act": "Energy",  # gas / nuclear
        "Disclosures Delegated Act - General": "Financial & Insurance",
        "EU Taxonomy - General": None,
        "Taxonomy-Alignment Reporting": "Financial & Insurance",
        "Taxonomy-Eligibility reporting (part 1)": "Financial & Insurance",
        "Taxonomy-Eligibility reporting (part 2)": "Financial & Insurance",
    }
    return MAPPING.get(section_name)


def parse_taxonomy_faqs_sector_aware(md_path: str | Path) -> List[FAQChunk]:
    """Parse FAQs and attach a 'sectors' list to every chunk."""
    chunks = parse_taxonomy_faqs(md_path)
    for chunk in chunks:
        # Fine-grained sector extraction from Q&A text
        sectors = _extract_sectors(chunk.question + " " + chunk.answer)
        # Fallback: use top-level section name for non-Climate-Delegated-Act chunks
        if not sectors:
            fallback = _classify_by_section(chunk.section)
            if fallback:
                sectors = {fallback}
        chunk.sectors = sorted(sectors) if sectors else ["General"]
    return chunks


@lru_cache
def load_documents_sector_aware() -> List[Document]:
    """FAQ entries as LangChain documents with sector metadata."""
    chunks = parse_taxonomy_faqs_sector_aware(FAQ_PATH)
    return [
        Document(
            page_content=f"Section: {chunk.section}\nQuestion: {chunk.question}\nAnswer: {chunk.answer}",
            metadata={
                "section": chunk.section,
                "question": chunk.question,
                "chunk_id": chunk.id,
                "sectors": chunk.sectors,
            },
        )
        for chunk in chunks
    ]
