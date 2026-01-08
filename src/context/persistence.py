"""
Context persistence logic.

Handles persisting and loading the application's context state
(selected tables) to/from ChromaDB for session continuity.
"""

import json
import streamlit as st
import pandas as pd

from src.database import get_chroma_client
from .manager import get_context, set_ctx_if_ready


def persist_current_context_marker() -> None:
    """
    Persists the list of currently selected tables to ChromaDB.
    
    Saves the selected_tables list to the app_state collection
    for recovery on next app restart.
    """
    try:
        client = get_chroma_client()
        coll = client.get_or_create_collection(name="app_state")
        ctx_id = "current_context"
        
        # Serialize the list of tables to JSON
        tables_json = json.dumps(st.session_state.selected_tables)
        
        metadata = {
            "tables_json": tables_json,
            "version": "2.0"  # versioning for future compatibility
        }
        
        # Create human-readable doc string
        table_names = [
            f"{t['project']}.{t['dataset']}.{t['table']}"
            for t in st.session_state.selected_tables
        ]
        doc = f"Current context tables: {', '.join(table_names)}"

        existing = coll.get(ids=[ctx_id], include=['metadatas', 'documents'])
        if existing['ids']:
            coll.update(ids=[ctx_id], documents=[doc], metadatas=[metadata])
        else:
            coll.add(ids=[ctx_id], documents=[doc], metadatas=[metadata])
        
    except Exception as e:
        st.warning(f"Could not persist app state to ChromaDB: {e}")


def load_persisted_context_to_session() -> None:
    """
    Load persisted context from ChromaDB (if any) into st.session_state.
    
    Attempts to restore previously selected tables and their schemas
    from the app_state collection. Falls back to loading from schema_context
    if BigQuery access fails.
    """
    if st.session_state.ctx_set:
        return

    try:
        from src.database import get_table_schema
        
        client = get_chroma_client()
        coll = client.get_or_create_collection(name="app_state")
        existing = coll.get(ids=["current_context"], include=['metadatas', 'documents'])
        
        if not existing['ids']:
            return
            
        meta = existing['metadatas'][0]
        tables_json = meta.get('tables_json')
        
        # Backward compatibility for single-table context
        if not tables_json:
            project = meta.get('project')
            dataset = meta.get('dataset')
            table = meta.get('table')
            description = meta.get('description', '')
            if all([project, dataset, table]):
                tables_list = [{
                    "project": project,
                    "dataset": dataset,
                    "table": table,
                    "description": description
                }]
            else:
                return
        else:
            try:
                tables_list = json.loads(tables_json)
            except json.JSONDecodeError:
                return

        loaded_tables = []
        loaded_schemas = {}
        last_schema_df = pd.DataFrame()

        for t_info in tables_list:
            project = t_info.get('project')
            dataset = t_info.get('dataset')
            table = t_info.get('table')
            description = t_info.get('description', '')
            
            if not all([project, dataset, table]):
                continue

            table_fqn = f"{project}.{dataset}.{table}"
            schema_found = False
            
            # Try to fetch schema from BQ
            try:
                schema = get_table_schema(project, dataset, table)
                if not schema.empty:
                    loaded_schemas[table_fqn] = schema
                    last_schema_df = schema
                    schema_found = True
            except Exception:
                pass
            
            # Fallback to Chroma if BQ failed
            if not schema_found:
                contexts = get_context(table)
                if contexts['ids']:
                    cols = []
                    for md in contexts.get('metadatas', []):
                        col_name = md.get('column_name')
                        desc = md.get('description', '')
                        cols.append({
                            'table_catalog': project,
                            'table_schema': dataset,
                            'table_name': table,
                            'column_name': col_name,
                            'data_type': 'UNKNOWN',
                            'column_description': desc
                        })
                    if cols:
                        loaded_schemas[table_fqn] = pd.DataFrame(cols)
                        last_schema_df = loaded_schemas[table_fqn]
                        schema_found = True

            if schema_found:
                loaded_tables.append(t_info)

        if loaded_tables:
            st.session_state.selected_tables = loaded_tables
            st.session_state.schemas = loaded_schemas
            st.session_state.schema_df = last_schema_df
            st.session_state.context_source = 'persisted'
            set_ctx_if_ready()
            st.session_state.auth = True
            print(f"Loaded {len(loaded_tables)} tables from persisted context.")

    except Exception as e:
        st.warning(f"Could not load persisted context: {e}")
        return
