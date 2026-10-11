# EU Taxonomy RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that answers questions based on the EU Taxonomy Navigator FAQs.

## Architecture

```
apps/
├── api/   FastAPI + LangChain + ChromaDB + OpenRouter
└── web/   Next.js 14 + Tailwind CSS
```

- **Backend**: Python FastAPI serving a RAG pipeline. The markdown FAQ is parsed into Q&A pairs (section → question → answer). Answers are generated with `openai/gpt-4o-mini`. All model calls go through [OpenRouter](https://openrouter.ai) (models overridable via `CHAT_MODEL` / `EMBEDDING_MODEL` in `.env`).
- **Frontend**: Next.js App Router with a single question bar and the answer below it. Proxies API calls to the FastAPI backend.

## Retrieval strategies

Every strategy feeds the same generation step (top 4 entries), so they differ only in how context is found. Pick one with `rag_type` on `/ask` (default `classic`, which the UI uses).

- **`classic`** — Q&A pairs embedded with `openai/text-embedding-3-small` and stored in ChromaDB; retrieval is a similarity search.
- **`vectorless`** — no embeddings. The LLM navigates the FAQ tree: it picks up to 3 sections from the table of contents, then picks the most relevant entries from those sections' question titles (2 extra LLM calls per question).
- **`sector_aware`** — the question is mapped to an economic sector (keyword match, LLM fallback), and the similarity search is filtered to that sector in a dedicated ChromaDB collection.
- **`hybrid`** — BM25 and dense retrieval fused with Reciprocal Rank Fusion into a 50-candidate pool, reranked locally with the `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder.

Set `use_rerank=true` to fetch 8 candidates and keep the 4 an LLM judge scores as most relevant (applies to `classic`, `vectorless` and `sector_aware`).

## Evaluation

`POST /evaluate` runs a pipeline on a golden set (`apps/api/rag/evaluate.py`) and scores it with [ragas](https://docs.ragas.io). The set mixes questions the FAQ answers (scored against the matching FAQ answer) with questions it does not cover, which the pipeline must decline:

- **retrieval_hit** — deterministic: was the target FAQ entry retrieved?
- **context_precision / context_recall** — ragas, LLM-judged against the reference answer.
- **faithfulness** — ragas: is the answer grounded in the retrieved context?
- **answer_relevancy** — ragas: does the answer address the question?
- **answer_rate** — share of answerable questions actually answered (catches over-cautious refusals).
- **refusal_rate** — share of unanswerable questions correctly declined. Refusals are classified by an LLM judge.

```bash
curl -X POST "http://localhost:8000/evaluate?rag_type=classic"
curl -X POST "http://localhost:8000/evaluate?rag_type=classic&use_rerank=true"
curl -X POST "http://localhost:8000/evaluate?rag_type=hybrid&prompt_type=expert"
```

### Results

Golden set of 20 questions (14 answerable, 6 unanswerable), `intermediate` prompt, `openai/gpt-4o-mini` for generation and judging. Bold marks the best score where pipelines differ.

| Pipeline | retrieval_hit | context_precision | context_recall | faithfulness | answer_relevancy | answer_rate | refusal_rate |
|---|---|---|---|---|---|---|---|
| Baseline (`classic`) | 1.000 | 0.966 | 1.000 | 0.888 | 0.793 | 0.929 | 1.000 |
| Baseline + LLM rerank (`classic`, `use_rerank`) | 1.000 | **0.980** | 1.000 | 0.889 | **0.799** | 0.929 | 1.000 |
| Hybrid (`hybrid`) | 1.000 | 0.974 | 1.000 | **0.929** | 0.744 | 0.929 | 1.000 |

All three pipelines retrieve the target entry for every answerable question and decline every unanswerable one; each answers 13 of the 14 answerable questions. Hybrid retrieval gives the most grounded answers (+0.04 faithfulness) at the cost of answer relevancy, while LLM reranking slightly sharpens context precision. Scores are LLM-judged, so expect small run-to-run variation.

#### Per question

Each cell reads **Baseline / Baseline + LLM rerank / Hybrid**. `retrieval_hit` and `context_recall` are 1.00 for every question and pipeline, so they are omitted. Questions 15–20 are not covered by the FAQ, so there is no reference to score against; all three pipelines declined them. A 0.00 on both faithfulness and answer relevancy (Q8 for Baseline and Baseline + LLM rerank, Q14 for Hybrid) is consistent with the pipeline declining an answerable question, since ragas scores noncommittal answers 0 for relevancy.

| # | Question | context_precision | faithfulness | answer_relevancy |
|---|---|---|---|---|
| 1 | Are the technical screening criteria under the Climate Delegated Act going to be tightened and reviewed periodically? | 1.00 / 1.00 / 0.92 | 1.00 / 1.00 / 1.00 | 0.92 / 0.94 / 0.94 |
| 2 | How frequently should forest management plans be renewed to remain compliant with the Taxonomy? | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.71 / 0.71 / 0.71 |
| 3 | If a parent company already reports consolidated Taxonomy figures, can its NFRD subsidiary skip its own disclosures? | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.74 / 0.73 / 0.72 |
| 4 | When does the 18-month completion window for individual CapEx measures start? | 0.81 / 0.81 / 0.81 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 5 | Can companies publish extra voluntary information alongside their mandatory Taxonomy-eligibility figures? | 1.00 / 1.00 / 1.00 | 1.00 / 0.86 / 1.00 | 0.98 / 1.00 / 0.98 |
| 6 | How should a bank that also holds a MiFID investment-firm licence report its Taxonomy-eligible activities? | 1.00 / 1.00 / 1.00 | 1.00 / 0.75 / 1.00 | 0.91 / 0.92 / 0.69 |
| 7 | What should a company disclose if none of its economic activities are Taxonomy-eligible? | 1.00 / 1.00 / 1.00 | 0.75 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 |
| 8 | Are derivative contracts part of the mandatory Taxonomy-eligibility reporting for financial firms? | 1.00 / 1.00 / 1.00 | 0.00 / 0.00 / 1.00 | 0.00 / 0.00 / 0.62 |
| 9 | Which new activities and technical criteria are introduced by the Complementary Climate Delegated Act? | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.75 / 0.75 / 0.75 |
| 10 | Why did the EU decide to label certain gas-related activities as substantially contributing to climate objectives? | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.81 / 0.64 / 0.45 |
| 11 | What advantages do companies gain from the disclosures delegated act? | 1.00 / 1.00 / 1.00 | 0.94 / 1.00 / 1.00 | 0.95 / 0.95 / 0.95 |
| 12 | How should financial companies treat their exposures to government entities for Taxonomy reporting? | 0.81 / 1.00 / 0.92 | 1.00 / 1.00 / 1.00 | 0.66 / 0.68 / 0.69 |
| 13 | How does the Taxonomy determine whether an economic activity counts as green? | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 / 1.00 | 0.91 / 0.93 / 0.93 |
| 14 | If a company's line of business is not listed in the EU Taxonomy, what does that imply? | 0.92 / 0.92 / 1.00 | 0.75 / 0.83 / 0.00 | 0.78 / 0.95 / 0.00 |
| 15 | What fines can national regulators impose on a company that knowingly misreports its Taxonomy KPIs? | — | — | — |
| 16 | What is the exact CO2 emission limit in grammes per kWh for electricity generation from geothermal energy under the Climate Delegated Act? | — | — | — |
| 17 | What was the total amount of green finance raised in the EU that was explicitly aligned with the Taxonomy in 2022? | — | — | — |
| 18 | By what exact deadline must EU Member States transpose the Taxonomy Regulation into national law? | — | — | — |
| 19 | Who won the 2018 FIFA World Cup? | — | — | — |
| 20 | What is the recipe for a traditional ratatouille? | — | — | — |

To reproduce, run from the repository root (writes `01_comparison_aggregated.csv` and a per-question breakdown, `02_per_question_all_methods.csv`):

```bash
source apps/api/.venv/bin/activate
python run_comparison.py
```

## Setup

### Prerequisites
- Python 3.11–3.12 and `pip`
- Node.js 20+ and `npm`
- OpenRouter API key

### 1. Backend

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then set OPENROUTER_API_KEY
uvicorn main:app --reload --port 8000
```

The first start embeds the FAQ into `apps/api/chroma_db/`; delete that folder to re-index after changing the data or embedding model.

### 2. Frontend

```bash
cd apps/web
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` and proxies `/api/*` requests to the backend at `http://localhost:8000`.

## Usage

- Open `http://localhost:3000`, type a question, and press **Enter**.

To compare retrieval strategies or prompt variants (`basic`, `intermediate`, `expert`), call the API directly:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What does DNSH mean?", "rag_type": "hybrid", "prompt_type": "expert"}'
```
