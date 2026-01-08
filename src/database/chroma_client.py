"""
In-memory vector store for semantic search.

Provides simple dictionary-based storage with numpy cosine similarity
for storing and retrieving table/column context with embeddings.
"""

import streamlit as st
import numpy as np
from typing import Optional


def _init_vector_store() -> None:
    """Initialize the in-memory vector store in session state if not exists."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = {
            "schema_context": {},
            "app_state": {}
        }


def get_context_collection() -> dict:
    """
    Get the schema_context collection (dictionary).

    Returns:
        dict: The schema_context storage dictionary
    """
    _init_vector_store()
    return st.session_state.vector_store["schema_context"]


def get_app_state_collection() -> dict:
    """
    Get the app_state collection (dictionary).

    Returns:
        dict: The app_state storage dictionary
    """
    _init_vector_store()
    return st.session_state.vector_store["app_state"]


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        float: Cosine similarity score (0 to 1, higher is more similar)
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def search_by_embedding(
    collection: dict,
    query_embedding: list[float],
    n_results: int = 5
) -> dict:
    """
    Search collection by embedding similarity.

    Args:
        collection: The collection dictionary to search
        query_embedding: Query embedding vector
        n_results: Number of results to return

    Returns:
        dict: Results with ids, metadatas, documents, distances
              (format matches ChromaDB for backward compatibility)
    """
    if not collection:
        return {
            "ids": [[]],
            "metadatas": [[]],
            "documents": [[]],
            "distances": [[]]
        }

    query_vec = np.asarray(query_embedding)

    # Calculate similarities for all items
    scored_items = []
    for item_id, item_data in collection.items():
        item_embedding = item_data.get("embedding")
        if item_embedding is not None:
            similarity = cosine_similarity(query_vec, item_embedding)
            # Convert similarity to distance (lower = better, matches ChromaDB)
            distance = 1.0 - similarity
            scored_items.append((item_id, item_data, distance))

    # Sort by distance (ascending - lower distance = more similar)
    scored_items.sort(key=lambda x: x[2])

    # Take top n_results
    top_results = scored_items[:n_results]

    # Format results to match ChromaDB query format
    return {
        "ids": [[item[0] for item in top_results]],
        "metadatas": [[item[1]["metadata"] for item in top_results]],
        "documents": [[item[1]["document"] for item in top_results]],
        "distances": [[item[2] for item in top_results]]
    }
