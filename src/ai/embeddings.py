"""
Embedding generation using Vertex AI.

Provides functions to generate embeddings for text using Google's
text embedding models.
"""

from .vertex_client import get_embedding_model


def get_embedding(text: str) -> list[float]:
    """
    Get embedding vector for a given text using Vertex AI.
    
    Args:
        text: Input text to embed
        
    Returns:
        list[float]: Embedding vector (typically 768 dimensions)
    """
    model = get_embedding_model()
    embeddings = model.get_embeddings([text])
    return embeddings[0].values
