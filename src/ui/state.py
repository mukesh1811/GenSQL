"""Session state initialization and management."""

import streamlit as st
import pandas as pd


def init_session_state() -> None:
    """
    Initializes all required session state variables.
    
    Sets default values for all session state keys used throughout the app.
    """
    ss = st.session_state
    
    # View state
    ss.setdefault("view", "analysis")  # analysis, context, view_context
    
    # Authentication and context
    ss.setdefault("auth", False)
    ss.setdefault("ctx_set", False)
    ss.setdefault("selected_tables", [])  # List of table info dicts
    ss.setdefault("schemas", {})  # Dict of table FQN -> schema DataFrame
    ss.setdefault("schema_df", pd.DataFrame())  # Legacy: kept for backward compatibility
    
    # BigQuery pickers state
    ss.setdefault("pick_project", None)
    ss.setdefault("pick_dataset", None)
    ss.setdefault("pick_table", None)
    
    # State for "Upload Schema" tab
    ss.setdefault("smpl_tbl_desc", "")
    ss.setdefault("upload_schema_df", pd.DataFrame())
    
    # State for "Pick BQ Table" tab
    ss.setdefault("bq_table_desc", "")
    ss.setdefault("bq_schema_df", pd.DataFrame())
    
    # State for chat messages and flow control
    ss.setdefault("messages", [])
    ss.setdefault("current_plan_approved", False)
    ss.setdefault("context_source", None)  # 'upload', 'bq', or 'persisted'
