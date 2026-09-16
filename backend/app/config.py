from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
class Settings(BaseSettings):
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-2-preview"
    gemini_rerank_model: str = "gemini-2.5-flash"
    data_dir: Path = Path("./data")
    max_retries: int = 2
    top_k_bm25: int = 30; top_k_vector: int = 30; top_k_rerank: int = 8
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")
settings = Settings()
