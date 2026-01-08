"""
Context persistence logic.

Handles persisting and loading the application's context state
within session state. Note: Data does not persist across app restarts.
"""

import json
import streamlit as st
import pandas as pd

from src.database import get_app_state_collection
from .manager import get_context, set_ctx_if_ready


def persist_current_context_marker() -> None:
    """
    Persists the list of currently selected tables to session state.

    Note: This only persists within the current session.
    Data is lost on app restart.
    """
    try:
        app_state = get_app_state_collection()

        # Serialize the list of tables to JSON
        tables_json = json.dumps(st.session_state.selected_tables)

        # Create human-readable doc string
        table_names = [
            f"{t['project']}.{t['dataset']}.{t['table']}"
            for t in st.session_state.selected_tables
        ]
        doc = f"Current context tables: {', '.join(table_names)}"

        app_state["current_context"] = {
            "document": doc,
            "metadata": {
                "tables_json": tables_json,
                "version": "2.0"
            }
        }

    except Exception as e:
        st.warning(f"Could not persist app state: {e}")


def load_persisted_context_to_session() -> None:
    """
    Load persisted context from session state (if any).

    Note: Since we use session state, this will only work within
    the same session. Context does not persist across app restarts.
    """
    if st.session_state.ctx_set:
        return

    try:
        from src.database import get_table_schema

        app_state = get_app_state_collection()

        if "current_context" not in app_state:
            return

        meta = app_state["current_context"].get("metadata", {})
        tables_json = meta.get('tables_json')

        if not tables_json:
            return

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

            # Fallback to in-memory store if BQ failed
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
            print(f"Loaded {len(loaded_tables)} tables from session context.")

    except Exception as e:
        st.warning(f"Could not load persisted context: {e}")
        return
