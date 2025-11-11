"""ChromaDB persistence utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_chroma_client() -> ClientAPI:
    """Return a cached Chroma persistent client."""

    settings = get_settings()
    base_path = Path(settings.chroma_directory)
    base_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(base_path),
        settings=Settings(persist_directory=str(base_path), anonymized_telemetry=False),
    )


def get_collection(name: str):
    """Shortcut to fetch or create a collection."""

    return get_chroma_client().get_or_create_collection(name=name)


def upsert_document(collection_name: str, document_id: str, document: str, embedding: list[float], metadata: dict[str, Any]):
    """Upsert a document into a Chroma collection."""

    collection = get_collection(collection_name)
    existing = collection.get(ids=[document_id], include=["metadatas"])
    if existing["ids"]:
        collection.update(ids=[document_id], documents=[document], embeddings=[embedding], metadatas=[metadata])
    else:
        collection.add(ids=[document_id], documents=[document], embeddings=[embedding], metadatas=[metadata])


def delete_document(collection_name: str, document_id: str) -> None:
    """Delete a document from a collection if present."""

    collection = get_collection(collection_name)
    collection.delete(ids=[document_id])


def query_similar(collection_name: str, embedding: list[float], n_results: int = 5):
    """Return similarity search results."""

    collection = get_collection(collection_name)
    return collection.query(query_embeddings=[embedding], n_results=n_results, include=["metadatas", "documents", "distances"])
