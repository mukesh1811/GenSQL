"""
Context manager for table/column metadata.

Handles saving, retrieving, and managing table and column context
in an in-memory vector store with embeddings for semantic search.
"""

import streamlit as st
import pandas as pd
import numpy as np

from src.database import get_context_collection, get_bq_client, get_app_state_collection, search_by_embedding
from src.ai import get_embedding


def save_context(
    table_id: str,
    column_name: str,
    description: str,
    project: str | None = None,
    dataset: str | None = None
) -> None:
    """
    Save or update context in the in-memory vector store with embeddings.

    Args:
        table_id: Table identifier
        column_name: Column name
        description: Column description
        project: Optional GCP project ID
        dataset: Optional dataset ID
    """
    collection = get_context_collection()
    context_id = f"{table_id}_{column_name}"

    # Generate embedding for the description
    embedding = get_embedding(description)

    # Prepare combined text for document (for better semantic search)
    combined_text = f"Table: {table_id}\nColumn: {column_name}\nDescription: {description}"

    # Build metadata dict with optional project/dataset
    metadata = {
        "table_id": table_id,
        "column_name": column_name,
        "description": description
    }
    if project:
        metadata["project"] = project
    if dataset:
        metadata["dataset"] = dataset

    # Store in dictionary (add or update is the same operation)
    collection[context_id] = {
        "id": context_id,
        "document": combined_text,
        "embedding": np.asarray(embedding),
        "metadata": metadata
    }


def persist_schema_to_chroma(
    project: str,
    dataset: str,
    table: str,
    schema_df: pd.DataFrame,
    table_description: str | None = None
) -> None:
    """
    Persist a full schema (table + columns) into the vector store.

    For each column in schema_df, saves context with embeddings.

    Args:
        project: GCP project ID
        dataset: Dataset ID
        table: Table name
        schema_df: DataFrame with columns including column_name and column_description
        table_description: Optional table-level description
    """
    table_id = table  # Keep backward compatibility

    for _, row in schema_df.iterrows():
        col = row.get('column_name')
        # Prefer explicitly provided column description
        desc = row.get('column_description') if row.get('column_description') is not None else ''
        final_desc = str(desc) if str(desc).strip() else (table_description or '')

        if not col:
            continue

        try:
            save_context(
                table_id=table_id,
                column_name=col,
                description=final_desc,
                project=project,
                dataset=dataset
            )
        except Exception as e:
            # Don't raise—log and continue to avoid breaking UI flow
            st.warning(f"Failed to save context for {table_id}.{col}: {e}")


def get_context(table_id: str = None, column_name: str = None) -> dict:
    """
    Retrieve context from the in-memory vector store.

    Args:
        table_id: Optional table ID filter
        column_name: Optional column name filter

    Returns:
        dict: Results with ids, metadatas, documents, embeddings
    """
    collection = get_context_collection()

    if table_id is None:
        # Get all contexts
        if not collection:
            return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}

        return {
            "ids": list(collection.keys()),
            "metadatas": [item["metadata"] for item in collection.values()],
            "documents": [item["document"] for item in collection.values()],
            "embeddings": [item["embedding"].tolist() if hasattr(item["embedding"], 'tolist')
                          else item["embedding"] for item in collection.values()]
        }

    if column_name is None:
        # Get all contexts for a specific table
        filtered = {k: v for k, v in collection.items()
                   if v["metadata"].get("table_id") == table_id}

        if not filtered:
            return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}

        return {
            "ids": list(filtered.keys()),
            "metadatas": [item["metadata"] for item in filtered.values()],
            "documents": [item["document"] for item in filtered.values()],
            "embeddings": [item["embedding"].tolist() if hasattr(item["embedding"], 'tolist')
                          else item["embedding"] for item in filtered.values()]
        }

    # Get specific context
    context_id = f"{table_id}_{column_name}"
    if context_id not in collection:
        return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}

    item = collection[context_id]
    return {
        "ids": [context_id],
        "metadatas": [item["metadata"]],
        "documents": [item["document"]],
        "embeddings": [item["embedding"].tolist() if hasattr(item["embedding"], 'tolist')
                      else item["embedding"]]
    }


def search_similar_contexts(query: str, n_results: int = 5) -> dict:
    """
    Search for similar contexts using semantic similarity.

    Args:
        query: Search query text
        n_results: Number of results to return

    Returns:
        dict: Query results with metadatas, documents, distances
    """
    collection = get_context_collection()
    query_embedding = get_embedding(query)

    results = search_by_embedding(collection, query_embedding, n_results)

    return results


def delete_context(table_id: str, column_name: str) -> None:
    """
    Delete a specific context from the in-memory store.

    Args:
        table_id: Table identifier
        column_name: Column name
    """
    collection = get_context_collection()
    context_id = f"{table_id}_{column_name}"

    if context_id in collection:
        del collection[context_id]


def set_ctx_if_ready() -> None:
    """Marks context as set if at least one table with schema is present."""
    has_tables = bool(st.session_state.selected_tables)
    has_schemas = bool(st.session_state.schemas) or not st.session_state.schema_df.empty
    st.session_state.ctx_set = has_tables and has_schemas


def set_context(
    project: str,
    dataset: str,
    table: str,
    description: str,
    schema_df: pd.DataFrame,
    source: str,
    append: bool = False
) -> None:
    """
    Sets the context, either by replacing existing or appending to it.

    Args:
        project: GCP project ID
        dataset: Dataset ID
        table: Table name
        description: Table description
        schema_df: Schema DataFrame
        source: Source of context ('upload', 'bq', etc.)
        append: If True, append to existing context; if False, replace
    """
    from .persistence import persist_current_context_marker

    if not all([project, dataset, table]) or schema_df.empty:
        st.error("Cannot set context with incomplete information.")
        return

    table_fqn = f"{project}.{dataset}.{table}"
    new_entry = {
        "project": project,
        "dataset": dataset,
        "table": table,
        "description": description
    }

    if not append:
        # Replace mode: clear existing
        st.session_state.selected_tables = [new_entry]
        st.session_state.schemas = {table_fqn: schema_df}
        st.session_state.schema_df = schema_df  # legacy support
    else:
        # Append mode
        # Check if already exists to avoid duplicates
        exists = any(
            t['project'] == project and t['dataset'] == dataset and t['table'] == table
            for t in st.session_state.selected_tables
        )
        if not exists:
            st.session_state.selected_tables.append(new_entry)
            st.session_state.schemas[table_fqn] = schema_df
            st.session_state.schema_df = schema_df

    st.session_state.context_source = source
    set_ctx_if_ready()

    # Persist schema & descriptions to vector store
    try:
        persist_schema_to_chroma(project, dataset, table, schema_df, description)
        persist_current_context_marker()
    except Exception as e:
        st.warning(f"Failed to persist schema: {e}")

    action = "Added" if append else "Set"
    st.toast(f"{action} context: `{project}.{dataset}.{table}`", icon="🧠")


def add_table_to_context(
    project: str,
    dataset: str,
    table: str,
    description: str,
    schema_df: pd.DataFrame,
    source: str
) -> None:
    """Adds a table to the existing context."""
    set_context(project, dataset, table, description, schema_df, source, append=True)


def remove_table_from_context(index: int) -> None:
    """
    Removes a table from context by index.

    Args:
        index: Index in selected_tables list
    """
    from .persistence import persist_current_context_marker

    if 0 <= index < len(st.session_state.selected_tables):
        removed = st.session_state.selected_tables.pop(index)
        table_fqn = f"{removed['project']}.{removed['dataset']}.{removed['table']}"
        st.session_state.schemas.pop(table_fqn, None)

        # If we removed the last table, clear legacy schema_df
        if not st.session_state.selected_tables:
            st.session_state.schema_df = pd.DataFrame()
        else:
            last = st.session_state.selected_tables[-1]
            last_fqn = f"{last['project']}.{last['dataset']}.{last['table']}"
            st.session_state.schema_df = st.session_state.schemas.get(
                last_fqn, pd.DataFrame()
            )

        set_ctx_if_ready()
        persist_current_context_marker()
        st.toast(f"Removed `{removed['table']}` from context.")
        st.rerun()


def clear_context() -> None:
    """Clears all context and chat history from the session."""
    st.session_state.selected_tables = []
    st.session_state.schemas = {}
    st.session_state.schema_df = pd.DataFrame()
    st.session_state.ctx_set = False
    st.session_state.context_source = None
    st.session_state.messages = []

    # Clear the in-memory stores
    try:
        schema_collection = get_context_collection()
        schema_collection.clear()

        app_state = get_app_state_collection()
        if "current_context" in app_state:
            del app_state["current_context"]
    except Exception as e:
        st.warning(f"Could not clear app state: {e}")

    st.toast("Context cleared.", icon="🗑️")
    st.rerun()
