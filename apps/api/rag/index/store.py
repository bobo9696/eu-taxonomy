"""Vector store setup using ChromaDB + OpenAI embeddings."""

from pathlib import Path

from langchain_chroma import Chroma

from rag.config import get_embeddings
from rag.index.methods.basic import load_documents

API_DIR = Path(__file__).parent.parent.parent
_VECTOR_STORE: Chroma | None = None


def get_vector_store(persist_dir: Path = API_DIR / "chroma_db") -> Chroma:
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    embeddings = get_embeddings()

    if (persist_dir / "chroma.sqlite3").exists():
        _VECTOR_STORE = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            collection_name="taxonomy_faqs",
        )
        return _VECTOR_STORE

    _VECTOR_STORE = Chroma.from_documents(
        documents=load_documents(),
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="taxonomy_faqs",
    )
    return _VECTOR_STORE
