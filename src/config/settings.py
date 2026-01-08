"""
Configuration settings for F.R.I.D.A.Y application.

Uses environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self):
        # GCP Configuration
        self.GCP_PROJECT_ID: Optional[str] = os.getenv("GCP_PROJECT_ID")
        self.GCP_REGION: str = os.getenv("GCP_REGION", "us-central1")
        self.VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")

        # Model Configuration
        self.GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.GEMINI_FALLBACK_MODEL: str = os.getenv(
            "GEMINI_FALLBACK_MODEL", "gemini-1.5-flash"
        )
        self.EMBEDDING_MODEL_PRIMARY: str = os.getenv(
            "EMBEDDING_MODEL_PRIMARY", "text-embedding-005"
        )
        self.EMBEDDING_MODEL_FALLBACK: str = os.getenv(
            "EMBEDDING_MODEL_FALLBACK", "gemini-embedding-001"
        )

        # Application Configuration
        self.APP_NAME: str = os.getenv("APP_NAME", "F.R.I.D.A.Y")
        self.APP_ICON: str = os.getenv("APP_ICON", "✨")
        self.ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")

        # Logging Configuration
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        # Feature Flags
        self.ENABLE_ANALYTICS: bool = os.getenv("ENABLE_ANALYTICS", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        self.ENABLE_COST_TRACKING: bool = os.getenv(
            "ENABLE_COST_TRACKING", "false"
        ).lower() in ("true", "1", "yes")

        # Cache Configuration
        self.CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default

        # BigQuery Configuration
        self.BQ_COST_PER_TB: float = float(
            os.getenv("BQ_COST_PER_TB", "5.0")
        )  # $5 per TB

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"

    def __repr__(self) -> str:
        """String representation of settings."""
        return f"<Settings environment={self.ENVIRONMENT}>"


# Global settings instance
settings = Settings()
