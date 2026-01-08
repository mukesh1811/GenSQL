"""
Vertex AI client initialization.

Provides cached Vertex AI models (Gemini for chat, embedding models).
"""

import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import google.auth
from google.cloud import bigquery

from src.config import settings


def _get_active_gcp_project() -> str:
    """
    Returns the GCP project ID using the same ADC path used for BigQuery.
    
    Tries:
    1) bigquery.Client().project (matches your schema/table access)
    2) google.auth.default() as fallback
    
    Returns:
        str: Active GCP project ID
        
    Raises:
        RuntimeError: If project ID cannot be determined
    """
    try:
        return bigquery.Client().project
    except Exception:
        _, project_id = google.auth.default()
        if not project_id:
            raise RuntimeError("Could not determine active GCP project from ADC.")
        return project_id


def safe_vertex_init():
    """
    Idempotent Vertex AI initialization using the same project as your BQ client.
    
    Uses settings.GCP_PROJECT_ID if provided, otherwise auto-detects from ADC.
    
    Raises:
        RuntimeError: If Vertex AI initialization fails
    """
    project_id = settings.GCP_PROJECT_ID or _get_active_gcp_project()
    try:
        vertexai.init(project=project_id, location=settings.VERTEX_LOCATION)
    except Exception as e:
        raise RuntimeError(
            f"Vertex init failed for project '{project_id}' "
            f"in '{settings.VERTEX_LOCATION}': {e}"
        )


@st.cache_resource(show_spinner="Initializing AI...")
def get_model() -> GenerativeModel:
    """
    Initializes Vertex AI and returns a Gemini model instance.
    
    Tries the primary model first, falls back to the fallback model
    if initialization fails.
    
    Returns:
        GenerativeModel: Initialized Gemini model
    """
    safe_vertex_init()
    
    # Try primary model first
    try:
        return GenerativeModel(settings.GEMINI_MODEL)
    except Exception:
        # Fallback to older/alternative model
        return GenerativeModel(settings.GEMINI_FALLBACK_MODEL)


@st.cache_resource(show_spinner="Initializing Embedding Model...")
def get_embedding_model() -> TextEmbeddingModel:
    """
    Initializes and returns a Text Embedding model instance.
    
    Tries multiple model candidates and returns the first one that works.
    
    Returns:
        TextEmbeddingModel: Initialized embedding model
        
    Raises:
        RuntimeError: If all embedding models fail to initialize
    """
    safe_vertex_init()
    
    candidates = [
        settings.EMBEDDING_MODEL_PRIMARY,
        settings.EMBEDDING_MODEL_FALLBACK,
    ]
    
    last_err = None
    for model_id in candidates:
        try:
            model = TextEmbeddingModel.from_pretrained(model_id)
            # Warmup to fail fast if there are scope/API issues
            _ = model.get_embeddings(["warmup"])
            return model
        except Exception as e:
            last_err = e
            continue
    
    raise RuntimeError(
        f"Failed to initialize an embedding model ({', '.join(candidates)}). "
        f"Last error: {last_err}"
    )
