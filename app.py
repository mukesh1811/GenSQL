# gcloud auth application-default login
# gcloud auth application-default set-quota-project learning-prj-id
# gcloud auth application-default set-quota-project aeo-supplychain-datamart-prod



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
    """Initializes and returns a BigQuery client."""
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
    """Lists available GCP projects."""
    client = get_bq_client()
    # Uses Resource Manager via BigQuery client under the hood
    return [p.project_id for p in client.list_projects()]


@st.cache_data(show_spinner=False)
def list_datasets(project_id: str):
    """Lists datasets in a given project."""
    client = get_bq_client()
    return [d.dataset_id for d in client.list_datasets(project=project_id)]


@st.cache_data(show_spinner=False)
def list_tables(project_id: str, dataset_id: str):
    """Lists tables in a given dataset."""
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
        # Add the editable description column
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
# Page Configuration
# =========================
st.set_page_config(
    page_title="F.R.I.D.A.Y",
    page_icon="✨",
    layout="centered",
)

# =========================
# Session State Initialization
# =========================
def _init_state():
    """Initializes all required session state variables."""
    ss = st.session_state
    ss.setdefault("view", "analysis")
    ss.setdefault("auth", False)
    ss.setdefault("ctx_set", False)
    ss.setdefault("selected_tables", [])
    ss.setdefault("schema_df", pd.DataFrame())
    ss.setdefault("pick", {"project": None, "dataset": None, "table": None})
    # State for "Upload Schema" tab
    ss.setdefault("smpl_tbl_desc", "")
    ss.setdefault("upload_schema_df", pd.DataFrame())
    # State for "Pick BQ Table" tab
    ss.setdefault("bq_table_desc", "")
    ss.setdefault("bq_schema_df", pd.DataFrame())
    # State for chat messages and flow control
    ss.setdefault("messages", [])
    ss.setdefault("current_plan_approved", False)

_init_state()


# =========================
# Helper Functions
# =========================
def set_ctx_if_ready():
    """Marks context as set if a table and schema are present."""
    st.session_state.ctx_set = bool(st.session_state.selected_tables) and (
        not st.session_state.schema_df.empty
    )

def set_context(project: str, dataset: str, table: str, description: str, schema_df: pd.DataFrame):
    """Overwrites the main context with new table information."""
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
    """Clears all context and chat history from the session."""
    st.session_state.selected_tables = []
    st.session_state.schema_df = pd.DataFrame()
    st.session_state.ctx_set = False
    st.session_state.messages = []
    st.toast("Context cleared.", icon="🗑️")
    st.rerun()

def add_bq_table_to_context():
    """Sets the selected BQ table as the main context."""
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
    """Resets selections when the GCP project changes."""
    st.session_state.pick.update({"dataset": None, "table": None})
    st.session_state.bq_schema_df = pd.DataFrame()
    st.session_state.bq_table_desc = ""

def on_dataset_change():
    """Resets selections when the dataset changes."""
    st.session_state.pick.update({"table": None})
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
        st.session_state.upload_schema_df = pd.read_csv("data/sample_schema.csv", dtype=str)
        # Reset the description in the callback to avoid widget state errors
        st.session_state.smpl_tbl_desc = ""
    except FileNotFoundError:
        st.error("`data/sample_schema.csv` not found. Please create this file for the sample to work.")
        st.session_state.upload_schema_df = pd.DataFrame()

# --- Gemini Helper Functions ---
def enhance_table_description_llm():
    """Uses the LLM to generate a table description based on its name and schema."""
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
    """Uses the LLM to generate descriptions for all columns in a table."""
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

def _get_llm_context():
    """Builds the context string for the LLM prompts."""
    if not st.session_state.selected_tables or st.session_state.schema_df.empty:
        st.error("Context is not set. Cannot generate SQL.")
        return None

    table_info = st.session_state.selected_tables[0]
    schema_df = st.session_state.schema_df
    project = table_info['project']
    dataset = table_info['dataset']
    table = table_info['table']
    table_description = table_info.get('description', 'No description provided.')
    schema_str = schema_df[['column_name', 'data_type', 'column_description']].to_string(index=False)

    return f"""
**Table Context:**
- Fully Qualified Table Name: `{project}.{dataset}.{table}`
- Table Description: {table_description}
- Table Schema:
{schema_str}
"""

def generate_plan_llm(conversation_text: str):
    """Generates a plan based on the conversation history."""
    model = get_model()
    context_str = _get_llm_context()
    if not context_str:
        return "Error: Context not set."

    prompt = f"""
You are an expert Google BigQuery SQL analyst. Your task is to create a step-by-step plan to answer the user's question based on the provided table context and conversation history.

{context_str}

**Conversation History:**
{conversation_text}

**Instructions:**
1. Analyze the latest user request in the context of the full conversation.
2. Generate a new, concise, step-by-step plan. If a previous plan exists, refine it based on the user's latest feedback.
3. The plan should be clear and easy to understand.
4. If the question cannot be answered using the schema, the plan must state why.
5. Output **only the plan text**, as a numbered or bulleted list. Do not include any preamble, titles, or markdown formatting.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"An error occurred while communicating with the AI model: {e}")
        return None

def generate_sql_from_plan_llm(conversation_text: str, approved_plan: str):
    """Generates SQL from an approved plan and conversation history."""
    model = get_model()
    context_str = _get_llm_context()
    if not context_str:
        return "Error: Context not set."
        
    prompt = f"""
You are an expert Google BigQuery SQL writer. Your task is to write a single, syntactically correct Google BigQuery SQL query based on the approved plan and table context.

{context_str}

**Full Conversation History (for context):**
{conversation_text}

**Approved Plan:**
---
{approved_plan}
---

**Instructions:**
1. Adhere strictly to the approved plan to write the query.
2. Write a single Google BigQuery SQL query.
3. If the plan states that the question is unanswerable, output the text "Unable to generate".
4. Output **only the SQL code**, without any preamble, explanation, or markdown formatting such as ```sql.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"An error occurred while communicating with the AI model: {e}")
        return None

def run_bigquery_query(sql_query: str) -> pd.DataFrame | None:
    """Executes a BigQuery SQL query and returns the result as a DataFrame."""
    try:
        client = get_bq_client()
        with st.spinner("Executing query in BigQuery..."):
            query_job = client.query(sql_query)
            results = query_job.to_dataframe()
        return results
    except GoogleAPIError as e:
        st.error(f"BigQuery Error: {e.message}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while running the query: {e}")
        return None

def suggest_chart_llm(user_prompt: str, sql_query: str, result_df: pd.DataFrame):
    """Analyzes a query result and suggests a suitable chart type."""
    if result_df.empty or len(result_df.columns) < 2:
        return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}

    model = get_model()
    data_preview = result_df.head().to_string()
    
    prompt = f"""
You are a data visualization expert. Based on the user's question, the SQL query, and a preview of the result data, determine the most suitable type of chart to visualize the answer.

**User's Question:** "{user_prompt}"

**SQL Query:**
```sql
{sql_query}
```

**Data Preview (first 5 rows):**
```
{data_preview}
```

**Instructions:**
1.  Analyze the data's structure (column names, number of columns, data types). A numeric column is generally required for the y-axis. A categorical or temporal column is best for the x-axis.
2.  Consider the user's intent (e.g., comparison, trend over time, distribution).
3.  Choose one of the following chart types: 'bar', 'line', 'area', 'scatter', or 'none'.
4.  If you choose a chart type other than 'none', you MUST identify the best columns for the 'x_axis' and 'y_axis' from the data preview.
5.  Provide a descriptive title for the chart.
6.  Return a single JSON object with the following keys: "chart_type", "x_axis", "y_axis", "title". If no chart is suitable, `chart_type` must be "none".

**JSON Output:**
"""
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        
        try:
            chart_info = json.loads(json_str)
            if "chart_type" in chart_info:
                return chart_info
            return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}
        except json.JSONDecodeError:
            return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}
            
    except Exception:
        return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}

# =========================
# Context Page UI
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
                # Clear description when new file is uploaded
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
            prj, dtset, tbl = df_editor['table_catalog'].unique()[0], df_editor['table_schema'].unique()[0], df_editor['table_name'].unique()[0]
            st.write(f"`{prj}.{dtset}.{tbl}`")

            c1, c2 = st.columns([9, 2])
            c1.text_area("Table Description", key="smpl_tbl_desc", placeholder="Add custom description for this table", label_visibility="collapsed")
            c2.button("Enhance", icon="🪄", help="A.I. will populate the table description for you", on_click=enh_smpl_tbl_desc, args=[tbl])

            st.divider()
            st.caption("**Table Schema**")
            edited_schema = st.data_editor(
                df_editor, hide_index=True,
                column_config={"column_description": st.column_config.TextColumn("Column Description", help="Describe the purpose of this column")},
                disabled=["table_catalog", "table_schema", "table_name", "column_name", "data_type"],
                key="schema_editor",
            )
            st.session_state.upload_schema_df = edited_schema

            c3, c4 = st.columns(2)
            if c3.button("Enhance Schema", icon="🪄", help="A.I. will populate the column description for you", key="enhance_upload_schema"):
                st.session_state.upload_schema_df = pd.read_csv("data/sample_schema_with_desc.csv")
                st.rerun()
            if c4.button("Set as Context", icon="🧠"):
                set_context(prj, dtset, tbl, st.session_state.smpl_tbl_desc, edited_schema)
                st.session_state.upload_schema_df = pd.DataFrame()
                # st.session_state.smpl_tbl_desc = ""
                st.rerun()

    with tab_pick:
        st.subheader("1. Authenticate with Google")
        c1, c2 = st.columns([2, 5], vertical_alignment="center", gap="small")
        if c1.button("Login to Google", icon="🔑", use_container_width=True):
            st.session_state.auth = True
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

            project = st.selectbox("GCP Project", [""] + projects, placeholder="Select a project", key="pick_project", on_change=on_project_change)
            st.session_state.pick["project"] = project or None

            datasets = list_datasets(project) if project else []
            dataset = st.selectbox("Dataset", [""] + datasets, placeholder="Select a dataset", key="pick_dataset", on_change=on_dataset_change)
            st.session_state.pick["dataset"] = dataset or None

            tables = list_tables(project, dataset) if project and dataset else []
            table = st.selectbox("Table", [""] + tables, placeholder="Select a table", key="pick_table")
            st.session_state.pick["table"] = table or None

            if all(st.session_state.pick.values()):
                if st.session_state.bq_schema_df.empty or st.session_state.bq_schema_df['table_name'].iloc[0] != table:
                    st.session_state.bq_schema_df = get_table_schema(project, dataset, table)
                    st.session_state.bq_table_desc = ""
                    st.rerun()

                st.divider()
                st.subheader("3. Preview & Enhance Context")
                c1, c2 = st.columns([9, 2])
                c1.text_area("Table Description", key="bq_table_desc", placeholder="Add custom description for this table", label_visibility="collapsed")
                c2.button("Enhance", icon="🪄", help="A.I. will populate table description", on_click=enhance_table_description_llm, key="enhance_bq_tbl_desc")

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
                    c3, c4 = st.columns(2)
                    c3.button("Enhance Schema", icon="🪄", help="A.I. will populate column descriptions", on_click=enhance_column_descriptions_llm, key="enhance_bq_cols")
                    c4.button("Set as Context", icon="🧠", on_click=add_bq_table_to_context)

    st.divider()
    st.caption("**Current Context:**")
    if not st.session_state.selected_tables:
        st.info("No context is set.")
    else:
        row = st.session_state.selected_tables[0]
        c1, c2 = st.columns([6, 1])
        with c1:
            st.write(f"• `{row['project']}.{row['dataset']}.{row['table']}`")
            if row.get("description"): st.caption(row['description'])
        c2.button("Clear Context", key="clear_ctx_btn", on_click=clear_context)

    st.divider()
    if st.button("Return", icon="⬅️", use_container_width=True):
        st.session_state.view = "analysis"
        st.rerun()


# =========================
# Analysis Page UI
# =========================
def build_llm_conversation_text(messages_list):
    """Formats the chat history for the LLM prompt."""
    history = []
    for msg in messages_list:
        role = "User" if msg['role'] == 'user' else "Assistant"
        content = msg.get('content', '') or msg.get('plan_text', '')
        history.append(f"{role}: {content}")
    return "\n".join(history)

def render_default_page():
    if not st.session_state.ctx_set:
        st.warning("Context not set. Please set context in the sidebar to proceed.", icon="⚠️")
        return

    st.markdown("<h2 style='text-align:center;margin-top:0;'>🧐 What are you analyzing today?</h2>", unsafe_allow_html=True)
    st.write("")

    # Find the index of the last message that is an unapproved plan
    last_plan_idx = -1
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        msg = st.session_state.messages[i]
        if msg.get("type") == "plan" and not msg.get("approved", False):
            last_plan_idx = i
            break
            
    # Display all messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["display_content"])

            if msg.get("type") == "sql":
                if "query_result" in msg:
                    st.caption("Query Result:")
                    st.dataframe(msg["query_result"])
                    
                    if msg.get("chart_created"):
                        st.caption("Chart:")
                        chart_info = msg["chart_suggestion"]
                        chart_df = msg["full_query_result"] # Use full result for charting
                        
                        # Verify columns exist before trying to chart
                        if chart_info["x_axis"] in chart_df.columns and chart_info["y_axis"] in chart_df.columns:
                            # Streamlit charts often work best with the x-axis set as the index
                            chart_df_indexed = chart_df.set_index(chart_info["x_axis"])
                            chart_type = chart_info["chart_type"]
                            
                            st.write(f"**{chart_info.get('title', 'Generated Chart')}**")

                            if chart_type == "bar": st.bar_chart(chart_df_indexed[[chart_info["y_axis"]]])
                            elif chart_type == "line": st.line_chart(chart_df_indexed[[chart_info["y_axis"]]])
                            elif chart_type == "area": st.area_chart(chart_df_indexed[[chart_info["y_axis"]]])
                            # Scatter doesn't use the index, so pass x and y
                            elif chart_type == "scatter": st.scatter_chart(chart_df, x=chart_info["x_axis"], y=chart_info["y_axis"])
                        else:
                            st.error("Chart generation failed: columns suggested by AI were not found in the result.")

                    elif "chart_suggestion" in msg and msg["chart_suggestion"]["chart_type"] != "none":
                        chart_info = msg["chart_suggestion"]
                        chart_type_str = chart_info['chart_type'].capitalize()
                        if st.button(f"📊 Create {chart_type_str} Chart", key=f"create_chart_{i}"):
                            st.session_state.messages[i]["chart_created"] = True
                            st.rerun()

                elif msg.get("sql_text") and "Unable to generate" not in msg["sql_text"]:
                    if st.button("🚀 Run SQL", key=f"run_sql_{i}"):
                        result_df = run_bigquery_query(msg["sql_text"])
                        if result_df is not None:
                            st.session_state.messages[i]["full_query_result"] = result_df # Store full result
                            
                            # Determine the preview dataframe (head(10))
                            if len(result_df) > 10:
                                st.warning(f"Displaying the first 10 of {len(result_df)} rows.")
                                st.session_state.messages[i]["query_result"] = result_df.head(10)
                            else:
                                st.session_state.messages[i]["query_result"] = result_df
                            
                            # Suggest chart based on the full result
                            with st.spinner("Analyzing result for chart suggestion..."):
                                original_prompt = ""
                                for j in range(i, -1, -1):
                                    if st.session_state.messages[j]["role"] == "user":
                                        original_prompt = st.session_state.messages[j]["content"]
                                        break
                                chart_suggestion = suggest_chart_llm(original_prompt, msg["sql_text"], result_df)
                                if chart_suggestion:
                                    st.session_state.messages[i]["chart_suggestion"] = chart_suggestion
                            st.rerun()

    # If the last message is an unapproved plan, show the Approve button
    if last_plan_idx != -1 and last_plan_idx == len(st.session_state.messages) - 1:
        st.session_state.current_plan_approved = False
        if st.button("✅ Approve Plan"):
            st.session_state.messages[last_plan_idx]["approved"] = True
            
            with st.chat_message("assistant"):
                with st.spinner("Generating SQL from approved plan..."):
                    approved_plan = st.session_state.messages[last_plan_idx]["plan_text"]
                    history_for_sql = st.session_state.messages[:last_plan_idx + 1]
                    conversation_text = build_llm_conversation_text(history_for_sql)
                    
                    sql_code = generate_sql_from_plan_llm(conversation_text, approved_plan)

                    if sql_code:
                        if "Unable to generate" in sql_code:
                            display_content = "⚠️ As per the plan, I am unable to generate the SQL for this request."
                        else:
                            display_content = f"**Generated SQL:**\n```sql\n{sql_code}\n```"

                        st.session_state.messages.append({
                            "role": "assistant", "type": "sql",
                            "sql_text": sql_code, "display_content": display_content
                        })
            st.rerun()

    # Chat input logic
    is_mid_conversation = last_plan_idx != -1 and last_plan_idx == len(st.session_state.messages) - 1
    placeholder = "Suggest an edit to the plan, or approve it." if is_mid_conversation else "What are you analyzing today?"
    
    if prompt := st.chat_input(placeholder):
        st.session_state.messages.append({"role": "user", "display_content": prompt, "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                conversation_text = build_llm_conversation_text(st.session_state.messages)
                new_plan_text = generate_plan_llm(conversation_text)

                if new_plan_text:
                    display_content = f"**Here is the proposed plan:**\n\n{new_plan_text}"
                    st.session_state.messages.append({
                        "role": "assistant", "type": "plan",
                        "plan_text": new_plan_text, "display_content": display_content,
                        "approved": False
                    })
        st.rerun()


# =========================
# Sidebar UI
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

if st.sidebar.button("Set/Change Context", icon="🧠", use_container_width=True):
    st.session_state.view = "context"
    st.rerun()

with st.sidebar.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
1. **Authenticate** with Google to enable access to your data warehouse.
2. **Set context**: choose a **single table** via schema upload or the BQ picker.
3. **Ask**: Describe what you want to analyze. The AI will propose a plan.
4. **Refine**: Chat with the AI to edit the plan until you're happy.
5. **Approve**: Click "Approve Plan" to generate the final SQL query.
6. **Run**: Execute the generated SQL against BigQuery and see the results.
7. **Visualize**: If applicable, click "Create Chart" to see a visual representation of your data.
""")

# =========================
# Main Router
# =========================
if st.session_state.view == "context":
    render_ctx_page()
else:
    render_default_page()