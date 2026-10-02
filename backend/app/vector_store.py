from functools import lru_cache
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .models import Chunk


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(settings.data_dir / "chroma"), settings=ChromaSettings(anonymized_telemetry=False))


def _collection(contract_id: str):
    return _client().get_or_create_collection(
        name=f"contract_{contract_id.replace('-', '_')}",
        metadata={"hnsw:space": "cosine"},
    )


def save_chunks(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    if not chunks or not vectors:
        return
    collection = _collection(chunks[0].contract_id)
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=vectors,
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {"page": chunk.page_start, "page_end": chunk.page_end, "clause": chunk.clause, "heading": chunk.heading}
            for chunk in chunks
        ],
    )


def search(contract_id: str, vector: list[float], limit: int) -> list[str]:
    try:
        result = _collection(contract_id).query(query_embeddings=[vector], n_results=limit)
    except Exception:
        return []
    return result.get("ids", [[]])[0]