"""
ChromaDB client for vector storage and retrieval.

Provides persistent ChromaDB client for storing and retrieving
table/column context with embeddings.
"""

import streamlit as st
import chromadb
from chromadb.config import Settings
from pathlib import Path

from src.config import settings as app_settings


@st.cache_resource(show_spinner="Initializing ChromaDB...")
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Initialize and return a ChromaDB client with local persistence.
    
    The persist directory is configured via settings (environment variable
    or default to ./chroma_db).
    
    Returns:
        chromadb.PersistentClient: Initialized ChromaDB client
    """
    persist_dir = app_settings.CHROMA_PERSIST_DIR
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            persist_directory=str(persist_dir),
            anonymized_telemetry=False
        )
    )


@st.cache_resource
def get_context_collection():
    """
    Get or create the ChromaDB collection for storing schema context.
    
    Returns:
        chromadb.Collection: The schema_context collection
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name="schema_context")


def get_app_state_collection():
    """
    Get or create the ChromaDB collection for storing app state.
    
    This collection is used for persisting selected tables and other
    application-level metadata.
    
    Returns:
        chromadb.Collection: The app_state collection
    """
    client = get_chroma_client()
    return client.get_or_create_collection(name="app_state")
