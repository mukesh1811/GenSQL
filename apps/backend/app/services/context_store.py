"""Context persistence and retrieval logic."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from app.services import chroma, vertex

SCHEMA_COLLECTION = "schema_context"
STATE_COLLECTION = "app_state"


def _context_document(table_id: str, column_name: str, description: str) -> str:
    return f"Table: {table_id}\nColumn: {column_name}\nDescription: {description}"


def persist_schema(project: str, dataset: str, table: str, schema_df: pd.DataFrame, table_description: str | None = None) -> None:
    """Persist schema metadata and embeddings into ChromaDB."""

    table_id = table
    for _, row in schema_df.iterrows():
        column_name = row.get("column_name")
        if not column_name:
            continue
        description = row.get("column_description")
        description = str(description) if description is not None else ""
        final_desc = description or (table_description or "")
        embedding = vertex.embed_text(final_desc)
        metadata = {
            "table_id": table_id,
            "column_name": column_name,
            "description": final_desc,
            "project": project,
            "dataset": dataset,
        }
        chroma.upsert_document(
            collection_name=SCHEMA_COLLECTION,
            document_id=f"{table_id}_{column_name}",
            document=_context_document(table_id, column_name, final_desc),
            embedding=embedding,
            metadata=metadata,
        )

    chroma.upsert_document(
        collection_name=STATE_COLLECTION,
        document_id="current_context",
        document=f"Current context: {project}.{dataset}.{table}",
        embedding=vertex.embed_text(f"{project}.{dataset}.{table}"),
        metadata={
            "project": project,
            "dataset": dataset,
            "table": table,
            "description": table_description or "",
        },
    )


def load_context(table_id: str | None = None, column_name: str | None = None):
    """Fetch stored context metadata."""

    collection = chroma.get_collection(SCHEMA_COLLECTION)
    if table_id is None:
        return collection.get(include=["metadatas", "documents", "embeddings"])
    if column_name is None:
        return collection.get(where={"table_id": table_id}, include=["metadatas", "documents", "embeddings"])
    return collection.get(ids=[f"{table_id}_{column_name}"], include=["metadatas", "documents", "embeddings"])


def delete_context(table: str, column: str) -> None:
    chroma.delete_document(SCHEMA_COLLECTION, f"{table}_{column}")


def load_current_context_marker() -> dict | None:
    collection = chroma.get_collection(STATE_COLLECTION)
    data = collection.get(ids=["current_context"], include=["metadatas"])
    if not data["ids"]:
        return None
    return data["metadatas"][0]


def search_contexts(query: str, n_results: int = 5):
    embedding = vertex.embed_text(query)
    results = chroma.query_similar(SCHEMA_COLLECTION, embedding, n_results=n_results)
    formatted: list[dict] = []
    for metadata, distance, document in zip(
        results.get("metadatas", []), results.get("distances", []), results.get("documents", [])
    ):
        formatted.append({
            "metadata": metadata,
            "distance": distance,
            "document": document,
        })
    return formatted


def dataframe_from_schema_records(records: Iterable[dict]) -> pd.DataFrame:
    """Convert schema records into a DataFrame."""

    return pd.DataFrame(list(records))
