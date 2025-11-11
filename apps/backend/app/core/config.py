from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(default="FRIDAY Analytics API", description="Application display name")
    api_prefix: str = Field(default="/api", description="Prefix for API routes")
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://localhost:3000"], description="Allowed CORS origins")
    chroma_directory: str = Field(default="chroma_db", description="Directory for persisting ChromaDB collections")
    vertex_location: str = Field(default="us-central1", description="Vertex AI region")

    class Config:
        env_file = ".env"
        env_prefix = "FRIDAY_"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
