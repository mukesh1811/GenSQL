"""Vertex AI helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from app.core.config import get_settings
from app.services.gcp import get_active_project


class VertexInitializationError(RuntimeError):
    """Raised when Vertex AI initialization fails."""


def _ensure_vertex_initialized() -> None:
    settings = get_settings()
    project = get_active_project()
    try:
        vertexai.init(project=project, location=settings.vertex_location)
    except Exception as exc:  # pragma: no cover - library-specific errors
        raise VertexInitializationError(
            f"Failed to initialize Vertex AI for project '{project}' in '{settings.vertex_location}': {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_generative_model() -> GenerativeModel:
    """Return the default Gemini model instance."""

    _ensure_vertex_initialized()
    preferred_models: Iterable[str] = ("gemini-2.5-flash", "gemini-1.5-flash")
    last_exc: Exception | None = None
    for model_name in preferred_models:
        try:
            return GenerativeModel(model_name)
        except Exception as exc:  # pragma: no cover - network/API dependent
            last_exc = exc
            continue
    raise VertexInitializationError(f"Could not initialize a generative model: {last_exc}")


@lru_cache(maxsize=1)
def get_embedding_model() -> TextEmbeddingModel:
    """Return the default text embedding model instance."""

    _ensure_vertex_initialized()
    candidates: Iterable[str] = ("text-embedding-005", "gemini-embedding-001")
    last_exc: Exception | None = None
    for name in candidates:
        try:
            model = TextEmbeddingModel.from_pretrained(name)
            _ = model.get_embeddings(["healthcheck"])  # fail fast
            return model
        except Exception as exc:  # pragma: no cover
            last_exc = exc
            continue
    raise VertexInitializationError(f"Could not initialize embedding model: {last_exc}")


def embed_text(text: str) -> list[float]:
    """Generate an embedding for the given text."""

    model = get_embedding_model()
    embedding = model.get_embeddings([text])[0]
    return embedding.values


def generate_content(prompt: str) -> str:
    """Utility wrapper for generative model."""

    model = get_generative_model()
    response = model.generate_content(prompt)
    return response.text.strip()
