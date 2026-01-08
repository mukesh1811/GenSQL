"""Context management page"""

import streamlit as st
import pandas as pd
from google.api_core.exceptions import GoogleAPIError, Forbidden

from src.utils.constants import SCHEMA_EXTRACTION_QRY
from src.database import (
    list_projects,
    list_datasets,
    list_tables,
    get_table_schema
)
from src.context.manager import (
    set_context,
    add_table_to_context,
    remove_table_from_context,
    clear_context,
    add_table_to_context as add_context_wrapper # Re-export or just use directly
)
from src.ai.column_enhancer import (
    enhance_table_description,
    enhance_column_descriptions
)

# --- Helper Functions (Callbacks) ---

def add_bq_table_to_context(append: bool = False):
    """Sets or adds the selected BQ table to context."""
    set_context(
        project=st.session_state.pick_project, 
        dataset=st.session_state.pick_dataset, 
        table=st.session_state.pick_table, 
        description=st.session_state.bq_table_desc,
        schema_df=st.session_state.bq_schema_df,
        source="bq",
        append=append
    )
    # Don't clear bq_schema_df, it's needed for the editor view
    st.rerun()

def on_project_change():
    """Resets selections when the GCP project changes."""
    st.session_state.pick_dataset = None 
    st.session_state.pick_table = None 
    st.session_state.bq_schema_df = pd.DataFrame()
    st.session_state.bq_table_desc = ""

def on_dataset_change():
    """Resets selections when the dataset changes."""
    st.session_state.pick_table = None 
    st.session_state.bq_schema_df = pd.DataFrame()
    st.session_state.bq_table_desc = ""

def enh_smpl_tbl_desc(tbl):
    """Provides a sample description for the Citibike table."""
    if tbl == "citibike_trips":
        tbl_desc = "The citibike_trips table contains log of all individual trips taken on the New York City (NYC) Citi Bike shared bicycle system"
    else:
        tbl_desc = "A sample table description."
    st.session_state.smpl_tbl_desc = tbl_desc

def load_sample_schema():
    """Callback to load the sample Citibike schema and reset description."""
    try:
        sample_data = {
            'table_catalog': ['bigquery-public-data', 'bigquery-public-data'],
            'table_schema': ['new_york_citibike', 'new_york_citibike'],
            'table_name': ['citibike_trips', 'citibike_trips'],
            'column_name': ['tripduration', 'starttime'],
            'data_type': ['INTEGER', 'TIMESTAMP'],
            'column_description': ['<IMP: Add a brief description>', '<IMP: Add a brief description>']
        }
        
        try:
            # Try to load from file first
            sample_df = pd.read_csv("data/sample_schema.csv", dtype=str)
        except FileNotFoundError:
            st.warning("`data/sample_schema.csv` not found. Using a minimal sample.")
            sample_df = pd.DataFrame(sample_data)
        
        st.session_state.upload_schema_df = sample_df
        st.session_state.smpl_tbl_desc = ""
    except Exception as e:
        st.error(f"Failed to load sample schema: {e}")
        st.session_state.upload_schema_df = pd.DataFrame()

# --- Main Render Function ---

def render_ctx_page():
    st.header("Set Context")
    st.caption("Set your context by uploading a schema file or picking a BigQuery table.")

    # Pre-populate edit view if changing context
    if (st.session_state.ctx_set and st.session_state.selected_tables): 
        table_info = st.session_state.selected_tables[0]
        current_schema_df = st.session_state.schema_df.copy()
        current_description = table_info.get('description', '')
        context_source = st.session_state.get('context_source')
        
        if context_source == "upload" and st.session_state.upload_schema_df.empty:
            st.session_state.upload_schema_df = current_schema_df
            st.session_state.smpl_tbl_desc = current_description
            
        elif context_source == "bq" and st.session_state.bq_schema_df.empty:
            st.session_state.bq_schema_df = current_schema_df
            st.session_state.bq_table_desc = current_description
            if st.session_state.pick_project is None:
                st.session_state.pick_project = table_info['project']
            if st.session_state.pick_dataset is None:
                st.session_state.pick_dataset = table_info['dataset']
            if st.session_state.pick_table is None:
                st.session_state.pick_table = table_info['table']
            st.session_state.auth = True

    tab_upl, tab_pick = st.tabs(["Upload Schema", "Pick a BigQuery table"])

    with tab_upl:
        st.subheader("1. Pick Schema")
        st.write("**Upload your schema file (CSV)**")

        with st.expander("ℹ️ How to get schema for your table (BigQuery)", expanded=False):
            st.code(SCHEMA_EXTRACTION_QRY, language="sql")

        uploaded = st.file_uploader(
            "**Upload your schema file (CSV)**", type=["csv"], label_visibility="collapsed"
        )
        if uploaded:
            try:
                st.session_state.upload_schema_df = pd.read_csv(uploaded, dtype=str)
                st.session_state.smpl_tbl_desc = ""
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.session_state.upload_schema_df = pd.DataFrame()

        st.write("(**or**)")
        st.write("**Use sample schema**")
        st.button("NY Citibike trips table schema", icon="🚲", on_click=load_sample_schema)

        upload_schema_df = st.session_state.upload_schema_df
        if not upload_schema_df.empty:
            col, default_desc = "column_description", "<IMP: Add a brief description>"
            if col not in upload_schema_df.columns:
                upload_schema_df[col] = default_desc
            upload_schema_df[col] = upload_schema_df[col].astype("string").fillna(default_desc)
            st.session_state.upload_schema_df = upload_schema_df

        st.divider()
        st.subheader("2. Schema Preview")

        if st.session_state.upload_schema_df.empty:
            st.info("Load a schema to begin.")
            if st.session_state.selected_tables:
                 st.caption("**Current Context:**")
                 row = st.session_state.selected_tables[0]
                 st.write(f"• `{row['project']}.{row['dataset']}.{row['table']}`")
        else:
            df_editor = st.session_state.upload_schema_df
            try:
                prj = df_editor['table_catalog'].unique()[0]
                dtset = df_editor['table_schema'].unique()[0]
                tbl = df_editor['table_name'].unique()[0]
            except (KeyError, IndexError):
                st.error("Uploaded CSV is missing required columns: 'table_catalog', 'table_schema', 'table_name'.")
                st.session_state.upload_schema_df = pd.DataFrame()
                return

            st.write(f"`{prj}.{dtset}.{tbl}`")

            c1, c2 = st.columns([9, 2])
            c1.text_area("Table Description", key="smpl_tbl_desc", placeholder="Add custom description for this table", label_visibility="collapsed")
            c2.button("Enhance", 
                            icon="🪄", help="A.I. will populate the table description for you", 
                            on_click=enh_smpl_tbl_desc,
                            key="enhance_upload_tbl_desc",
                            args=[tbl])

            st.divider()
            st.caption("**Table Schema**")
            edited_schema = st.data_editor(
                df_editor, hide_index=True,
                column_config={"column_description": st.column_config.TextColumn("Column Description", help="Describe the purpose of this column")},
                disabled=["table_catalog", "table_schema", "table_name", "column_name", "data_type"],
                key="schema_editor",
            )
            st.session_state.upload_schema_df = edited_schema

            c3, c4, c5 = st.columns([1, 1, 1])
            if c3.button("Enhance Schema", icon="🪄", help="A.I. will populate the column description for you", key="enhance_upload_schema"):
                # Simplified enhance here or call a func? The original logic had fallback.
                # Since we don't have the LLM logic in this file, we can't easily reproduce the exact fallback logic 
                # without importing more stuff.
                # However, the user wants "Enhance Schema" for upload which relies on LLM.
                # The original code for upload enhancement relied on a local CSV fallback mostly?
                # Actually line 1237 in original app.py shows it loading data/sample_schema_with_desc.csv
                try:
                    sample_data_desc = {
                        'table_catalog': ['bigquery-public-data', 'bigquery-public-data'],
                        'table_schema': ['new_york_citibike', 'new_york_citibike'],
                        'table_name': ['citibike_trips', 'citibike_trips'],
                        'column_name': ['tripduration', 'starttime'],
                        'data_type': ['INTEGER', 'TIMESTAMP'],
                        'column_description': ['The duration of the trip in seconds.', 'The time the trip started.']
                    }
                    try:
                        sample_df_desc = pd.read_csv("data/sample_schema_with_desc.csv")
                    except FileNotFoundError:
                        st.warning("`data/sample_schema_with_desc.csv` not found. Using minimal enhanced sample.")
                        sample_df_desc = pd.DataFrame(sample_data_desc)
                        
                    st.session_state.upload_schema_df = sample_df_desc
                except Exception as e:
                    st.error(f"Failed to load enhanced sample: {e}")
                st.rerun()
            
            if c4.button("Set as Context", icon="🧠", help="Replace existing context with this table", key="set_context_upload_btn"):
                set_context(prj, dtset, tbl, st.session_state.smpl_tbl_desc, edited_schema, source="upload", append=False)
                st.session_state.upload_schema_df = pd.DataFrame()
                st.rerun()
                
            if c5.button("Add to Context", icon="➕", help="Add this table to existing context", key="add_context_upload_btn"):
                add_table_to_context(prj, dtset, tbl, st.session_state.smpl_tbl_desc, edited_schema, source="upload")
                st.session_state.upload_schema_df = pd.DataFrame()
                st.rerun()

    with tab_pick:
        st.subheader("1. Authenticate with Google")
        c1, c2 = st.columns([2, 5], vertical_alignment="center", gap="small")
        if c1.button("Login to Google", icon="🔑", use_container_width=True, disabled=st.session_state.auth):
            st.session_state.auth = True
            st.rerun()
        with c2:
            if st.session_state.auth: st.success("Login Successful.", icon="✅")
            else: st.info("Please login to access your data warehouse.", icon="ℹ️")

        st.subheader("2. Choose Project → Dataset → Table")
        if not st.session_state.auth:
            st.warning("Please authenticate first.", icon="⚠️")
        else:
            try:
                with st.spinner("Loading projects..."): projects = list_projects()
            except Forbidden:
                st.error("Permission denied. Ensure your account has access.")
                projects = []
            except GoogleAPIError as e:
                st.error(f"Error listing projects: {e}")
                projects = []
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                projects = []

            project = st.selectbox("GCP Project", [""] + projects, placeholder="Select a project", key="pick_project", on_change=on_project_change)

            project_val = st.session_state.pick_project
            datasets = list_datasets(project_val) if project_val else []
            dataset = st.selectbox("Dataset", [""] + datasets, placeholder="Select a dataset", key="pick_dataset", on_change=on_dataset_change)

            dataset_val = st.session_state.pick_dataset
            tables = list_tables(project_val, dataset_val) if project_val and dataset_val else []
            table = st.selectbox("Table", [""] + tables, placeholder="Select a table", key="pick_table")

            table_val = st.session_state.pick_table
            
            if all([project_val, dataset_val, table_val]):
                if st.session_state.bq_schema_df.empty or st.session_state.bq_schema_df['table_name'].iloc[0] != table_val:
                    st.session_state.bq_schema_df = get_table_schema(project_val, dataset_val, table_val)
                    st.session_state.bq_table_desc = ""
                    st.rerun()
                
                c1, c2 = st.columns([9, 2])
                c1.text_area("Table Description", key="bq_table_desc", placeholder="Add custom description for this table", label_visibility="collapsed")
                c2.button("Enhance", icon="🪄", 
                            help="A.I. will populate table description", 
                            on_click=enhance_table_description, 
                            key="enhance_bq_tbl_desc"
                            )

                if not st.session_state.bq_schema_df.empty:
                    st.divider()
                    st.caption("**Table Schema**")
                    edited_bq_schema = st.data_editor(
                        st.session_state.bq_schema_df, hide_index=True,
                        column_config={"column_description": st.column_config.TextColumn("Column Description", help="Describe this column")},
                        disabled=["table_catalog", "table_schema", "table_name", "column_name", "data_type"],
                        key="bq_schema_editor"
                    )
                    st.session_state.bq_schema_df = edited_bq_schema
                    c3, c4, c5 = st.columns([1, 1, 1])
                    c3.button("Enhance Schema", icon="🪄", help="A.I. will populate column descriptions", on_click=enhance_column_descriptions, key="enhance_bq_cols")
                    c4.button("Set as Context", icon="🧠", help="Replace existing context", on_click=add_bq_table_to_context, kwargs={"append": False}, key="set_context_bq_btn")
                    c5.button("Add to Context", icon="➕", help="Add to existing context", on_click=add_bq_table_to_context, kwargs={"append": True}, key="add_context_bq_btn")

    st.divider()
    st.divider()
    st.subheader(f"Current Context ({len(st.session_state.selected_tables)} tables)")
    if not st.session_state.selected_tables:
        st.info("No context is set.")
    else:
        for idx, row in enumerate(st.session_state.selected_tables):
            with st.container():
                c1, c2 = st.columns([8, 1])
                with c1:
                    st.markdown(f"**{idx+1}.** `{row['project']}.{row['dataset']}.{row['table']}`")
                    if row.get("description"): 
                        st.caption(row['description'])
                with c2:
                    st.button("❌", key=f"rm_ctx_{idx}", help="Remove table", on_click=remove_table_from_context, args=[idx])
                st.divider()
        
        if st.button("Clear All Context", key="clear_ctx_btn", type="primary"):
            clear_context()

    st.divider()
    if st.button("Return", icon="⬅️", use_container_width=True):
        st.session_state.view = "analysis"
        st.rerun()
