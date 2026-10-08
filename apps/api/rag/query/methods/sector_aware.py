"""Sector-aware retrieval: filter by economic sector before similarity search."""

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from rag.config import get_llm
from rag.index.store_sector_aware import get_vector_store_sector_aware

# ---------------------------------------------------------------------------
# Fast keyword-based sector classifier (no LLM call needed for common cases)
# ---------------------------------------------------------------------------
_SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Forestry": ["forest", "afforestation", "reforestation", "wood", "timber", "tree", "deforestation"],
    "Manufacturing": ["manufactur", "product", "industr", "steel", "cement", "aluminium", "plastics", "hydrogen", "chemical"],
    "Energy": ["electricity", "power", "renewable", "solar", "wind", "hydro", "geothermal", "biomass", "biogas", "fuel", "heat pump", "storage", "grid"],
    "Water & Waste": ["water", "waste", "sewage", "recycling", "treatment", "circular"],
    "Transport": ["transport", "vehicle", "rail", "road", "shipping", "aviation", "freight", "passenger", "maritime", "cycle"],
    "Construction": ["building", "construction", "renovation", "retrofit", "insulation", "HVAC"],
    "ICT": ["software", "data", "digital", "telecommunication", "programming", "computer", "network", "cloud"],
    "Professional Services": ["consultancy", "consultant", "engineering", "advice", "professional", "audit"],
    "Financial & Insurance": ["bank", "financial", "investment", "insurance", "credit", "fund", "KPI", "disclosure", "reporting", "NFRD", "CSRD"],
}

_ALL_SECTORS = sorted(_SECTOR_KEYWORDS.keys()) + ["General"]


class SectorClassification(BaseModel):
    sector: str = Field(
        description="The economic sector most relevant to the user's question",
        examples=_ALL_SECTORS,
    )


SECTOR_CLASSIFIER_PROMPT = """Classify the user's question into exactly one of the following economic sectors.
If the question does not clearly map to any sector, answer "General".

Available sectors:
{sectors}

Rules:
- Forestry: forest management, afforestation, reforestation, wood
- Manufacturing: steel, cement, aluminium, plastics, hydrogen, industrial processes
- Energy: electricity generation, renewables, solar, wind, geothermal, storage, grid
- Water & Waste: water treatment, sewage, waste management, recycling
- Transport: rail, road, shipping, aviation, freight, vehicle
- Construction: buildings, renovation, retrofit, HVAC, insulation
- ICT: software, data, telecom, computer programming, digital infrastructure
- Professional Services: consultancy, engineering, advisory, audit
- Financial & Insurance: banking, insurance, investment, reporting, KPIs, disclosures

Question: {question}

Sector:"""


def _classify_by_keywords(question: str) -> str | None:
    q = question.lower()
    scores: dict[str, int] = {}
    for sector, keywords in _SECTOR_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score:
            scores[sector] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def classify_question_sector(question: str) -> str:
    """Return the most relevant sector for a user question."""
    fast = _classify_by_keywords(question)
    if fast:
        return fast
    # Fallback: lightweight LLM call
    llm = get_llm()
    result = llm.with_structured_output(SectorClassification).invoke(
        SECTOR_CLASSIFIER_PROMPT.format(
            sectors="\n".join(f"- {s}" for s in _ALL_SECTORS),
            question=question,
        )
    )
    return result.sector if result.sector in _ALL_SECTORS else "General"


def sector_aware_retrieve(question: str, k: int = 4) -> list[Document]:
    """Retrieve docs, first filtering by the sector inferred from the question."""
    store = get_vector_store_sector_aware()
    sector = classify_question_sector(question)

    # 1. Try sector-filtered search
    if sector != "General":
        filtered = store.similarity_search(question, k=k, filter={"sectors": {"$contains": sector}})
        if len(filtered) >= k:
            return filtered[:k]

    # 2. Fallback to unrestricted search
    return store.similarity_search(question, k=k)
