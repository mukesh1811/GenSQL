# gcloud auth application-default login

import pandas as pd
import streamlit as st
import json

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, Forbidden
import google.auth

# Gemini / Vertex AI Imports
import vertexai
from vertexai.generative_models import GenerativeModel


@st.cache_resource(show_spinner=True)
def get_bq_client():
    # Uses ADC from `gcloud auth application-default login`
    return bigquery.Client()


@st.cache_resource(show_spinner="Initializing AI...")
def get_model():
    """Initializes Vertex AI and returns a Gemini model instance."""
    try:
        # Get project ID from Application Default Credentials
        _, project_id = google.auth.default()
        vertexai.init(project=project_id)
    except (google.auth.exceptions.DefaultCredentialsError, AttributeError):
        st.warning("Could not determine GCP project from credentials. Some AI features may not work.")
        # Fallback initialization
        vertexai.init()
    # Use the stable model identifier for the latest version
    return GenerativeModel("gemini-2.5-flash")


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


@st.cache_data(show_spinner="Fetching table schema...")
def get_table_schema(project_id: str, dataset_id: str, table_id: str) -> pd.DataFrame:
    """Fetches the schema for a given BigQuery table."""
    client = get_bq_client()
    query = f"""
        SELECT
            table_catalog,
            table_schema,
            table_name,
            column_name,
            data_type
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_id}'
    """
    try:
        df = client.query(query).to_dataframe()
        # Add the editable description column, matching the other tab
        default_desc = "<IMP: Add a brief description>"
        df["column_description"] = default_desc
        df["column_description"] = df["column_description"].astype("string")
        return df
    except Exception as e:
        st.error(f"Failed to fetch schema: {e}")
        return pd.DataFrame()


# =========================
# Constants
# =========================
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
    ss.setdefault("view", "analysis")
    ss.setdefault("auth", False)
    ss.setdefault("ctx_set", False)
    ss.setdefault("selected_tables", [])
    ss.setdefault("schema_df", pd.DataFrame())
    ss.setdefault("pick", {"project": None, "dataset": None, "table": None})
    # State for "Upload Schema" tab
    ss.setdefault("smpl_tbl_desc", "")
    ss.setdefault("upload_schema_df", pd.DataFrame()) # Holds the temporary schema for preview
    # State for "Pick BQ Table" tab
    ss.setdefault("bq_table_desc", "")
    ss.setdefault("bq_schema_df", pd.DataFrame())


_init_state()


# =========================
# Helpers
# =========================
def set_ctx_if_ready():
    """Mark context set if we have a selected table and a non-empty schema."""
    st.session_state.ctx_set = bool(st.session_state.selected_tables) and (
        not st.session_state.schema_df.empty
    )


def set_context(project: str, dataset: str, table: str, description: str, schema_df: pd.DataFrame):
    """Overwrites the current context with the provided table information."""
    if not all([project, dataset, table]) or schema_df.empty:
        st.error("Cannot set context with incomplete information.")
        return

    st.session_state.selected_tables = [
        {"project": project, "dataset": dataset, "table": table, "description": description}
    ]
    st.session_state.schema_df = schema_df
    set_ctx_if_ready()
    st.toast(f"Context set to `{project}.{dataset}.{table}`", icon="🧠")


def clear_context():
    """Clears all context from the session state."""
    st.session_state.selected_tables = []
    st.session_state.schema_df = pd.DataFrame()
    st.session_state.ctx_set = False
    st.toast("Context cleared.", icon="🗑️")
    st.rerun()


def add_bq_table_to_context():
    """Overwrites the main context with the selected BQ table and its schema."""
    set_context(
        project=st.session_state.pick["project"],
        dataset=st.session_state.pick["dataset"],
        table=st.session_state.pick["table"],
        description=st.session_state.bq_table_desc,
        schema_df=st.session_state.bq_schema_df
    )
    st.session_state.bq_schema_df = pd.DataFrame()
    st.session_state.bq_table_desc = ""
    st.rerun()


def on_project_change():
    """Reset downstream selections and temporary schema state when project changes."""
    st.session_state.pick.update({"dataset": None, "table": None})
    st.session_state.bq_schema_df = pd.DataFrame()
    st.session_state.bq_table_desc = ""


def on_dataset_change():
    """Reset downstream selections and temporary schema state when dataset changes."""
    st.session_state.pick.update({"table": None})
    st.session_state.bq_schema_df = pd.DataFrame()
    st.session_state.bq_table_desc = ""


def enh_smpl_tbl_desc():
    st.session_state.smpl_tbl_desc = "The citibike_trips table contains log of all individual trips taken on the New York City (NYC) Citi Bike shared bicycle system"


# --- Gemini Helper Functions ---
def enhance_table_description_llm():
    model = get_model()
    project, dataset, table = st.session_state.pick.values()
    schema_df = st.session_state.bq_schema_df

    if not all([project, dataset, table]) or schema_df.empty:
        st.warning("Please select a valid table first.")
        return

    schema_str = "\n".join([f"- {row.column_name} ({row.data_type})" for _, row in schema_df.iterrows()])
    prompt = f"""
    Based on the fully qualified table name `{project}.{dataset}.{table}` and its schema, please provide a concise, one-sentence description of what this table likely contains.

    Schema:
    {schema_str}

    Description:
    """
    with st.spinner("🪄 Enhancing table description..."):
        try:
            response = model.generate_content(prompt)
            st.session_state.bq_table_desc = response.text.strip()
        except Exception as e:
            st.error(f"AI enhancement failed: {e}")


def enhance_column_descriptions_llm():
    model = get_model()
    project, dataset, table = st.session_state.pick.values()
    schema_df = st.session_state.bq_schema_df.copy()

    if not all([project, dataset, table]) or schema_df.empty:
        st.warning("Please select a valid table first.")
        return

    cols_to_describe = schema_df[['column_name', 'data_type']].to_dict('records')
    prompt = f"""
    Given the table name `{project}.{dataset}.{table}`, provide a concise, one-line description for each of the following columns.
    Return the output as a simple JSON object where keys are the column names and values are the descriptions. Do not include any other text or markdown formatting.

    Columns:
    {cols_to_describe}

    JSON Output:
    """
    with st.spinner("🪄 Enhancing column descriptions..."):
        try:
            response = model.generate_content(prompt)
            json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()

            try:
                desc_dict = json.loads(json_str)
                descriptions = pd.Series(desc_dict)
                schema_df['column_description'] = schema_df['column_name'].map(descriptions).fillna(schema_df['column_description'])
                st.session_state.bq_schema_df = schema_df
                st.rerun()
            except json.JSONDecodeError:
                st.error("AI model returned an invalid JSON format. Please try again.")
                st.code(json_str, language="json")

        except Exception as e:
            st.error(f"AI enhancement failed: {e}")


# =========================
# Context Page
# =========================
def render_ctx_page():
    st.header("Set Context")
    st.caption("Set your context by uploading a schema file or picking a BigQuery table.")

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
                st.session_state.smpl_tbl_desc = "" # Reset description on new upload
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                st.session_state.upload_schema_df = pd.DataFrame()

        st.write("(**or**)")
        st.write("**Use sample schema**")
        if st.button("NY Citibike trips table schema", icon="🚲"):
            try:
                # A real app might download this, here we load from local
                st.session_state.upload_schema_df = pd.read_csv("data/sample_schema.csv", dtype=str)
                st.session_state.smpl_tbl_desc = "" # Reset description
            except FileNotFoundError:
                st.error("`data/sample_schema.csv` not found. Please create this file for the sample to work.")
                st.session_state.upload_schema_df = pd.DataFrame()


        # This is the robust sanitization step for the uploaded schema
        upload_schema_df = st.session_state.upload_schema_df
        if not upload_schema_df.empty:
            col, default_desc = "column_description", "<IMP: Add a brief description>"
            
            # If the description column doesn't exist, add it
            if col not in upload_schema_df.columns:
                upload_schema_df[col] = default_desc
            
            # Crucial Fix: Ensure the column is string type and fill NAs
            # This handles cases where the column exists but has empty values read as NaN
            upload_schema_df[col] = upload_schema_df[col].astype("string").fillna(default_desc)
            
            # Update the session state with the sanitized dataframe
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
            upload_schema_df_for_editor = st.session_state.upload_schema_df
            st.caption("**Preview of new context:**")
            prj, dtset, tbl = upload_schema_df_for_editor['table_catalog'].unique()[0], upload_schema_df_for_editor['table_schema'].unique()[0], upload_schema_df_for_editor['table_name'].unique()[0]
            st.write(f"`{prj}.{dtset}.{tbl}`")

            col_tbldesc, col_enhbtn = st.columns([9, 2], vertical_alignment="center")
            with col_tbldesc:
                st.text_area("Table Description", key="smpl_tbl_desc", placeholder="Add custom description for this table", label_visibility="collapsed")
            with col_enhbtn:
                st.button("Enhance", icon="🪄", width="stretch", help="A.I. will populate the table description for you", on_click=enh_smpl_tbl_desc)

            st.divider()
            st.caption("**Table Schema**")
            edited_schema = st.data_editor(
                upload_schema_df_for_editor, hide_index=True, width="stretch",
                column_config={"column_description": st.column_config.TextColumn("Column Description", help="Describe the purpose of this column")},
                disabled=["table_catalog", "table_schema", "table_name", "column_name", "data_type"],
                key="schema_editor",
            )
            st.session_state.upload_schema_df = edited_schema

            col_enh, col_memadd = st.columns(2)
            with col_enh:
                if st.button("Enhance Schema", 
                             icon="🪄", 
                             width="stretch", 
                             help="A.I. will populate the column description for you", 
                             key="enhance_upload_schema"):
                    st.session_state.upload_schema_df = pd.read_csv("data/sample_schema_with_desc.csv")
                    st.rerun()
            with col_memadd:
                if st.button("Set as Context", icon="🧠", width="stretch"):
                    set_context(
                        project=prj, dataset=dtset, table=tbl,
                        description=st.session_state.smpl_tbl_desc,
                        schema_df=edited_schema
                    )
                    st.session_state.upload_schema_df = pd.DataFrame() # Clear temp state
                    # st.session_state.smpl_tbl_desc = ""
                    st.rerun()

    with tab_pick:
        st.subheader("1. Authenticate with Google")
        c1, c2 = st.columns([2, 5], vertical_alignment="center", gap="small")
        with c1:
            if st.button("Login to Google", icon="🔑", width="stretch"):
                st.session_state.auth = True
        with c2:
            if st.session_state.auth: st.success("Login Successful.", icon="✅")
            else: st.info("Please login to access your data warehouse.", icon="ℹ️")

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
            st.session_state.pick["project"] = project if project else None

            datasets = list_datasets(project) if project else []
            dataset = st.selectbox("Dataset", options=[""] + datasets, index=0, help="Pick the dataset to work in.", key="pick_dataset", on_change=on_dataset_change)
            st.session_state.pick["dataset"] = dataset or None

            tables = list_tables(project, dataset) if project and dataset else []
            table = st.selectbox("Table", options=[""] + tables, index=0, help="Pick a table to analyze", key="pick_table")
            st.session_state.pick["table"] = table or None

            if all(st.session_state.pick.values()):
                if st.session_state.bq_schema_df.empty or st.session_state.bq_schema_df['table_name'].iloc[0] != table:
                    st.session_state.bq_schema_df = get_table_schema(project, dataset, table)
                    st.session_state.bq_table_desc = ""
                    st.rerun()

                st.divider()
                st.subheader("3. Preview & Enhance Context")
                col_tbldesc, col_enhbtn = st.columns([9, 2], vertical_alignment="center")
                with col_tbldesc:
                    st.text_area("Table Description", key="bq_table_desc", placeholder="Add custom description for this table", label_visibility="collapsed")
                with col_enhbtn:
                    st.button("Enhance", icon="🪄", width="stretch", help="A.I. will populate the table description for you", on_click=enhance_table_description_llm, key="enhance_bq_tbl_desc")

                if not st.session_state.bq_schema_df.empty:
                    st.divider()
                    st.caption("**Table Schema**")
                    edited_bq_schema = st.data_editor(
                        st.session_state.bq_schema_df, hide_index=True, width="stretch",
                        column_config={"column_description": st.column_config.TextColumn("Column Description", help="Describe the purpose of this column")},
                        disabled=["table_catalog", "table_schema", "table_name", "column_name", "data_type"],
                        key="bq_schema_editor"
                    )
                    st.session_state.bq_schema_df = edited_bq_schema
                    col_enh, col_memadd = st.columns(2)
                    with col_enh:
                        st.button("Enhance Schema", icon="🪄", width="stretch", help="A.I. will populate the column descriptions for you", on_click=enhance_column_descriptions_llm, key="enhance_bq_cols")
                    with col_memadd:
                        st.button("Set as Context", icon="🧠", width="stretch", on_click=add_bq_table_to_context)

        st.divider()
        st.caption("**Current Context:**")
        if not st.session_state.selected_tables:
            st.info("No context is set.")
        else:
            row = st.session_state.selected_tables[0]
            col_a, col_b = st.columns([6, 1])
            with col_a:
                st.write(f"• `{row['project']}.{row['dataset']}.{row['table']}`")
                if row.get("description"): st.caption(row['description'])
            with col_b:
                st.button("Clear Context", key="clear_ctx_btn", on_click=clear_context)

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

    with st.container():
        prompt = st.chat_input("Type your requirement and generate the SQL")
        if prompt:
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
        row = st.session_state.selected_tables[0]
        st.sidebar.write(f"**Table:** `{row['project']}.{row['dataset']}.{row['table']}`")
        if row.get("description"):
            st.sidebar.caption(row['description'])
    if not st.session_state.schema_df.empty:
        st.sidebar.caption("Schema: loaded")

if st.sidebar.button("Set Context", icon="🧠", width='stretch'):
    st.session_state.view = "context"
    st.rerun()

with st.sidebar.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
1. **Authenticate** with Google to enable access to your data warehouse.
2. **Set context**: choose a **single table** via schema upload or the BQ picker. Setting a new context will replace the old one.
3. **Ask**: Describe what you want to analyze.
4. **Review**: Check the SQL the assistant drafts for you.
5. **Iterate**: Refine your prompt or change the context table.
""")

# =========================
# Router
# =========================
if st.session_state.view == "context":
    render_ctx_page()
else:
    render_default_page()

