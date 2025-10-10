# gcloud auth application-default login

import pandas as pd
import streamlit as st

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, Forbidden

@st.cache_resource(show_spinner=True)
def get_bq_client():
    # Uses ADC from `gcloud auth application-default login`
    return bigquery.Client()

@st.cache_data(show_spinner=False)
def list_projects():
    client = get_bq_client()
    # Uses Resource Manager via BigQuery client under the hood
    return [p.project_id for p in client.list_projects()]

@st.cache_data(show_spinner=False)
def list_datasets(project_id: str):
    client = get_bq_client()
    return [d.dataset_id for d in client.list_datasets(project=project_id)]

@st.cache_data(show_spinner=False)
def list_tables(project_id: str, dataset_id: str):
    client = get_bq_client()
    return [t.table_id for t in client.list_tables(f"{project_id}.{dataset_id}")]

# =========================
# Constants / Dummy Catalog
# =========================
DUMMY_CATALOG = {
    "gen-prod": {
        "sales": ["orders", "customers", "order_items"],
        "supply_chain": ["shipments", "inventory", "suppliers"],
    },
    "gen-dev": {
        "playground": ["events", "users", "sessions"],
        "marketing": ["campaigns", "leads", "touchpoints"],
    },
}

SCHEMA_EXTRACTION_QRY = """
-- Copy & run in BigQuery to list the table schema
SELECT
  table_catalog,
  table_schema,
  table_name,
  column_name,
  data_type,
  column_description
FROM `<your-project-id>.<your_dataset_id>.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '<your_table_name>';
""".strip()


# =========================
# Page Configuration (keep first)
# =========================
st.set_page_config(
    page_title="F.R.I.D.A.Y",
    page_icon="✨",
    layout="centered",
)

# =========================
# Session Defaults / State
# =========================
def _init_state():
    ss = st.session_state
    ss.setdefault("view", "analysis")             # "analysis" | "context"
    ss.setdefault("auth", False)                  # Simulated auth
    ss.setdefault("ctx_set", False)               # Becomes True when a table is selected or schema uploaded
    ss.setdefault("selected_tables", [])          # List of dicts: [{"project":..., "dataset":..., "table":...}]
    ss.setdefault("schema_df", pd.DataFrame())    # Last loaded/edited schema
    ss.setdefault("pick", {"project": None, "dataset": None, "table": None})  # UI selections
    # ss.setdefault("smpl_tbl_desc" , "")
    # st.session_state.smpl_tbl_desc = ""

_init_state()


# =========================
# Helpers
# =========================
def enh_smpl_tbl_desc():
    st.session_state.smpl_tbl_desc = "The citibike_trips table contains log of all individual trips taken on the New York City (NYC) Citi Bike shared bicycle system"

def set_ctx_if_ready():
    """Mark context set if we have at least one selected table OR a non-empty schema."""
    st.session_state.ctx_set = bool(st.session_state.selected_tables) or (
        not st.session_state.schema_df.empty
    )

def add_selected_table(project, dataset, table):
    if not (project and dataset and table):
        return
    entry = {"project": project, "dataset": dataset, "table": table}
    if entry not in st.session_state.selected_tables:
        st.session_state.selected_tables.append(entry)
        set_ctx_if_ready()

def remove_selected_table(idx: int):
    if 0 <= idx < len(st.session_state.selected_tables):
        st.session_state.selected_tables.pop(idx)
        set_ctx_if_ready()

def get_datasets(project):
    return sorted(list(DUMMY_CATALOG.get(project, {}).keys()))

def get_tables(project, dataset):
    return sorted(DUMMY_CATALOG.get(project, {}).get(dataset, []))


# =========================
# Context Page
# =========================
def render_ctx_page():
    st.header("Set Context")
    st.caption("Set your context by uploading a schema file or picking a BigQuery table.")

    tab_upl, tab_pick = st.tabs(["Upload Schema", "Pick a BigQuery table"])

    # ---------- Upload Schema ----------
    with tab_upl:
        st.subheader("1. Pick Schema")
        st.write("**Upload your schema file (CSV)**")

        with st.expander("ℹ️ How to get schema for your table (BigQuery)", expanded=False):
            st.code(SCHEMA_EXTRACTION_QRY, language="sql")

        uploaded = st.file_uploader(
            "**Upload your schema file (CSV)**",
            type=["csv"],
            label_visibility="collapsed",
        )

        schema = st.session_state.schema_df.copy()

        if uploaded:
            try:
                schema = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        st.write("(**or**)")
        st.write("**Use sample schema**")
        if st.button("NY Citibike trips table schema", icon="🚲"):
            # Expect a local sample for demo; replace with your path or inline sample
            schema = pd.DataFrame(
                {
                    "column_name": ["ride_id", "started_at", "ended_at", "start_station_id", "end_station_id"],
                    "data_type": ["STRING", "TIMESTAMP", "TIMESTAMP", "STRING", "STRING"],
                }
            )
            schema = pd.read_csv("data/sample_schema.csv")

        # Ensure description column exists & editable
        col = "column_description"
        default_desc = "<IMP: Add a brief description>"
        if not schema.empty:
            if col not in schema.columns:
                schema[col] = default_desc
            schema[col] = schema[col].astype("string").fillna(default_desc)

        st.divider()
        st.subheader("2. Schema Preview")


        if schema.empty:
            st.info("No schema loaded yet.")
        else:
            st.caption("**Selected tables:**")

            prj = schema['table_catalog'].unique()[0]
            dtset = schema['table_schema'].unique()[0]
            tbl = schema['table_name'].unique()[0]

            
            st.write(f"`{prj}.{dtset}.{tbl}`")
            
            col_tbldesc , col_enhbtn = st.columns([9,2],
                                               vertical_alignment="center"
                                              )
            
            with col_tbldesc:
                st.text_area("Table Description",
                    # value=st.session_state.smpl_tbl_desc,
                    key="smpl_tbl_desc",
                    placeholder="Add custom description for this table",
                    label_visibility="collapsed",
                    height="content"
                )
            
            with col_enhbtn:
                st.button("Enhance", 
                          icon = "🪄" , 
                          width="stretch" , 
                          help="A.I. will populate the table description for you",
                          on_click=enh_smpl_tbl_desc
                        
                        )
            
            st.divider()
            st.caption("**Table Schema**")
            # Editable data editor for descriptions
            edited = st.data_editor(
                schema,
                hide_index=True,
                width="stretch",
                column_config={
                    "column_description": st.column_config.TextColumn(
                        "Column Description", help="Describe the purpose of this column"
                    )
                },
                disabled={c: True for c in schema.columns if c in ["column_name", "data_type"]},
                key="schema_editor",
            )
            st.session_state.schema_df = edited
            col_enh , col_memadd = st.columns(2)
            
            with col_enh:
                if st.button("Enhance",
                          icon = "🪄" ,
                          width="stretch" ,
                          help="A.I. will populate the column description for you"
                        ):
                    st.session_state.schema_df = pd.read_csv("data/sample_schema_with_desc.csv")
                    st.rerun()
            
            with col_memadd:
                if st.button("Add to memory" , icon="🧠",width="stretch"):
                    
                    st.session_state.pick["project"] = prj
                    st.session_state.pick["dataset"] = dtset
                    st.session_state.pick["table"] = tbl
            
                    add_selected_table(
                        st.session_state.pick["project"],
                        st.session_state.pick["dataset"],
                        st.session_state.pick["table"]
                    )
                    set_ctx_if_ready()
                    st.rerun()
            
            if st.session_state.ctx_set:
                st.success("Schema ready. You can now generate SQL.", icon="✅")

    # ---------- Pick a BigQuery table ----------
    with tab_pick:
        st.subheader("1. Authenticate with Google")
        c1, c2 = st.columns([2, 5], vertical_alignment="center", gap="small")
        with c1:
            if st.button("Login to Google", icon="🔑", width="stretch"):
                st.session_state.auth = True
        with c2:
            if st.session_state.auth:
                st.success("Login Successful. You can now access your data warehouse.", icon="✅")
            else:
                st.info("Please login to access your data warehouse.", icon="ℹ️")

        st.subheader("2. Choose Project → Dataset → Table")
        if not st.session_state.auth:
            st.warning("Please authenticate first.", icon="⚠️")
        else:

            # Projects
            try:
                with st.spinner("Loading projects..."):
                    projects = list_projects()
            except Forbidden:
                st.error("Permission denied while listing projects. Ensure your account has access.")
                projects = []
            except GoogleAPIError as e:
                st.error(f"Error listing projects: {e}")
                projects = []

            project = st.selectbox(
                "GCP Project",
                options=[""] + projects,
                index=0,
                help="Pick the GCP project to work in.",
                placeholder="Select a project",
                key="pick_project",
                on_change=lambda: st.session_state.pick.update({"dataset": None, "table": None}),
            )
            st.session_state.pick["project"] = project or None

            # Datasets
            datasets = []
            if project:
                try:
                    with st.spinner("Loading datasets..."):
                        datasets = list_datasets(project)
                except Forbidden as f:
                    st.error(f)
                    st.error(f"Permission denied listing datasets in project `{project}`.")
                except GoogleAPIError as e:
                    st.error(f"Error listing datasets: {e}")

            dataset = st.selectbox(
                "Dataset",
                options=[""] + datasets,
                index=0,
                help="Pick the dataset to work in.",
                key="pick_dataset",
                on_change=lambda: st.session_state.pick.update({"table": None}),
            )
            st.session_state.pick["dataset"] = dataset or None

            # Tables
            tables = []
            if project and dataset:
                try:
                    with st.spinner("Loading tables..."):
                        tables = list_tables(project, dataset)
                except Forbidden:
                    st.error(f"Permission denied listing tables in `{project}.{dataset}`.")
                except GoogleAPIError as e:
                    st.error(f"Error listing tables: {e}")

            table = st.selectbox(
                "Table",
                options=[""] + tables,
                index=0,
                help="Pick a table to analyze",
                key="pick_table",
            )
            st.session_state.pick["table"] = table or None

            can_add = all(st.session_state.pick.values())
            st.button(
                "Add table to selection",
                icon="➕",
                width="stretch",
                disabled=not can_add,
                on_click=lambda: add_selected_table(
                    st.session_state.pick["project"],
                    st.session_state.pick["dataset"],
                    st.session_state.pick["table"],
                ),
            )

            st.caption("**Selected tables:**")
            if not st.session_state.selected_tables:
                st.info("No tables selected yet.")
            else:
                for i, row in enumerate(st.session_state.selected_tables):
                    col_a, col_b = st.columns([6, 1])
                    with col_a:
                        st.write(f"• `{row['project']}.{row['dataset']}.{row['table']}`")
                    with col_b:
                        st.button("Remove", key=f"rm_{i}", on_click=remove_selected_table, args=(i,))

    st.divider()
    if st.button("Return", icon="⬅️", width="stretch"):
        st.session_state.view = "analysis"
        st.rerun()


# =========================
# Analysis Page
# =========================
def render_default_page():
    if not st.session_state.ctx_set:
        st.warning("Context not set. Please set context in the sidebar to proceed.", icon="⚠️")

    st.markdown("<h2 style='text-align:center;margin-top:0;'>🧐 What are you analyzing today?</h2>", unsafe_allow_html=True)
    st.write("")

    # Chat input (always visible at top)
    with st.container():
        prompt = st.chat_input("Type your requirement and generate the SQL")
        if prompt:
            # Here you could call your SQL generator later
            st.chat_message("user").write(prompt)
            st.chat_message("assistant").write("I'll generate SQL once the SQL engine is wired up. ✅")


# =========================
# Sidebar
# =========================
st.sidebar.title("F.R.I.D.A.Y")
st.sidebar.caption("AI-Powered SQL Analytics Assistant")

st.sidebar.title("Context")
if not st.session_state.ctx_set:
    st.sidebar.warning("Context is empty", icon="⚠️")
else:
    st.sidebar.success("Context set", icon="✅")
    if st.session_state.selected_tables:
        st.sidebar.caption("Tables:")
        for row in st.session_state.selected_tables:
            st.sidebar.write(f"- `{row['project']}.{row['dataset']}.{row['table']}`")
    if not st.session_state.schema_df.empty:
        st.sidebar.caption("Schema: loaded")

if st.sidebar.button("Set Context", icon="🧠", width='stretch'):
    st.session_state.view = "context"

with st.sidebar.expander("ℹ️ How it works", expanded=False):
    st.markdown(
        """
1. **Authenticate** with Google to enable access to your data warehouse.  
2. **Set context**: choose **Project → Dataset → Table** in *Set Context* or upload a CSV schema.  
3. **Ask** in *Generate SQL*: describe what you want to analyze.  
4. **Review SQL** the assistant drafts using your current selections.  
5. **Iterate**: refine your prompt or change selections; chat history persists for the session.
"""
    )

# =========================
# Router
# =========================
if st.session_state.view == "context":
    render_ctx_page()
else:
    render_default_page()


