"""FastAPI application for EU Taxonomy FAQ RAG chatbot."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from rag.query.chain import PromptType, RagType, ask_question
from rag.evaluate import evaluate_rag
from rag.index.store import get_vector_store


class AskRequest(BaseModel):
    question: str
    prompt_type: PromptType = "intermediate"
    rag_type: RagType = "classic"
    use_rerank: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    contexts: list[str] = []
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class ScoreRequest(BaseModel):
    question: str
    answer: str
    contexts: list[str]


class ScoreResponse(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None


class EvaluateResponse(BaseModel):
    rag_type: str
    prompt_type: str
    use_rerank: bool = False
    samples: list[dict]
    metrics: dict


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up vector store on startup
    print("Loading vector store...")
    get_vector_store()
    print("Vector store ready.")
    yield


app = FastAPI(title="EU Taxonomy RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "EU Taxonomy RAG API is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Ask a question to the RAG-powered EU Taxonomy chatbot."""
    result = ask_question(
        req.question,
        prompt_type=req.prompt_type,
        rag_type=req.rag_type,
        use_rerank=req.use_rerank,
    )
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        contexts=result["contexts"],
        retrieval_ms=result["retrieval_ms"],
        generation_ms=result["generation_ms"],
        total_ms=result["total_ms"],
    )


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    """Score a single response with ragas (faithfulness + answer relevancy)."""
    from rag.evaluate import score_single_response
    result = await score_single_response(req.question, req.answer, req.contexts)
    return ScoreResponse(**result)


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    rag_type: RagType = "classic",
    prompt_type: PromptType = "intermediate",
    use_rerank: bool = False,
):
    """Score a RAG pipeline with ragas on the golden set of FAQ questions."""
    result = await evaluate_rag(rag_type, prompt_type, use_rerank)
    return EvaluateResponse(**result)
