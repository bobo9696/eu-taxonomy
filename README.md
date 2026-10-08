# EU Taxonomy RAG Chatbot

A Retrieval-Augmented Generation (RAG) application that answers questions based on the EU Taxonomy Navigator FAQs.

## Architecture

```
apps/
├── api/   FastAPI + LangChain + ChromaDB + OpenRouter
└── web/   Next.js 14 + Tailwind CSS
```

- **Backend**: Python FastAPI serving a RAG pipeline. The markdown FAQ is parsed into Q&A pairs (section → question → answer). Answers are generated with `openai/gpt-4o-mini`. All model calls go through [OpenRouter](https://openrouter.ai) (models overridable via `CHAT_MODEL` / `EMBEDDING_MODEL` in `.env`).
- **Frontend**: Next.js App Router with a minimal chat UI. Proxies API calls to the FastAPI backend.

## Retrieval strategies

Both strategies feed the same generation step (top 4 entries), so they differ only in how context is found. Pick one with `rag_type` on `/ask` or in the UI.

- **`classic`** — Q&A pairs embedded with `openai/text-embedding-3-small` and stored in ChromaDB; retrieval is a similarity search.
- **`vectorless`** — no embeddings. The LLM navigates the FAQ tree: it picks up to 3 sections from the table of contents, then picks the most relevant entries from those sections' question titles (2 extra LLM calls per question).

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
curl -X POST "http://localhost:8000/evaluate?rag_type=vectorless&prompt_type=basic"
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

- Open `http://localhost:3000`, type a question, and click **Ask**.
- Toggle between **Basic** and **Improved** prompt variants to compare output quality.
- Sources are shown under each answer for provenance.
# eu-taxonomy
