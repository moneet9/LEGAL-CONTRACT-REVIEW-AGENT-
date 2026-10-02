from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    model = SentenceTransformer(settings.embedding_model)
    model.max_seq_length = settings.embedding_max_tokens
    return model


def embed(texts: list[str], progress_callback=None, batch_size: int | None = None) -> list[list[float]]:
    if not texts:
        return []
    model = _model()
    batch_size = batch_size or settings.embedding_batch_size
    batches = []
    for start in range(0, len(texts), batch_size):
        vectors = model.encode(texts[start:start + batch_size], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        batches.extend(np.asarray(vectors, dtype="float32").tolist())
        if progress_callback:
            progress_callback(min(start + batch_size, len(texts)), len(texts))
    values = np.asarray(batches, dtype="float32")
    if values.ndim != 2 or values.shape[1] != settings.embedding_dimension:
        raise RuntimeError(
            f"Embedding model returned {values.shape[1] if values.ndim == 2 else 'invalid'} dimensions; "
            f"expected {settings.embedding_dimension}."
        )
    return values.tolist()


@lru_cache(maxsize=128)
def embed_query(text: str) -> list[float]:
    return embed([text])[0]