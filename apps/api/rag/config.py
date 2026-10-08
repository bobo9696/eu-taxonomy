"""Model clients routed through OpenRouter's OpenAI-compatible API."""

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import AsyncOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")


def _api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return key


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0.0,
        api_key=_api_key(),
        base_url=OPENROUTER_BASE_URL,
        timeout=120.0,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=_api_key(),
        base_url=OPENROUTER_BASE_URL,
        # Send raw strings: OpenRouter does not accept pre-tokenized input.
        check_embedding_ctx_length=False,
    )


def get_async_client() -> AsyncOpenAI:
    """Raw OpenRouter client, used by the ragas evaluation metrics."""
    return AsyncOpenAI(api_key=_api_key(), base_url=OPENROUTER_BASE_URL, timeout=120.0)
