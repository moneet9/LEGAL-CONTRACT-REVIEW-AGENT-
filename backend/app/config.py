from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-flash-lite-latest"
    embedding_model: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    embedding_max_tokens: int = 256
    enable_local_embeddings: bool = True
    chunk_size_chars: int = 900
    chunk_overlap_chars: int = 120
    ollama_base_url: str = "http://localhost:11434"
    data_dir: Path = Path("./data")
    max_retries: int = 2
    review_context_chars: int = 50000
    top_k_bm25: int = 30; top_k_vector: int = 30; top_k_rerank: int = 8
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")
settings = Settings()
