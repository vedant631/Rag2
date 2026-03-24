from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Postgres
    postgres_dsn: str = "postgresql+asyncpg://rag:rag@localhost:5432/ragdb"

    # OpenSearch
    opensearch_url: str = "http://localhost:5700"
    opensearch_index: str = "documents-rag"
    opensearch_dims: int = 768  # matches embedding model output

    # Services
    vision_url: str = "http://localhost:5600"
    reranker_url: str = "http://localhost:5601"

    # LLM (final answer — can be same llama-server or separate)
    llm_url: str = "http://localhost:5600"
    llm_model: str = "qwen"

    # Pipeline
    top_k_retrieve: int = 20   # how many to pull from opensearch
    top_k_rerank: int = 5      # how many to keep after reranking

    class Config:
        env_file = ".env"


settings = Settings()