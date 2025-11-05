## auth
# gcloud auth login
# gcloud auth application-default login

## set project
# gcloud config set project aeo-supplychain-datamart-prod
# gcloud auth application-default set-quota-project aeo-supplychain-datamart-prod

## set project
# gcloud config set project learning-prj-id
# gcloud auth application-default set-quota-project learning-prj-id



import pandas as pd
import streamlit as st
import json

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError, Forbidden
import google.auth

# Gemini / Vertex AI Imports
import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import chromadb
from chromadb.config import Settings
import os
import numpy as np
from pathlib import Path


VERTEX_LOCATION = "us-central1"  # fixed region; no env vars

def _safe_vertex_init():
    """Idempotent Vertex init using the same project as your BQ client."""
    project_id = _get_active_gcp_project()
    try:
        vertexai.init(project=project_id, location=VERTEX_LOCATION)
    except Exception as e:
        raise RuntimeError(
            f"Vertex init failed for project '{project_id}' in '{VERTEX_LOCATION}': {e}"
        )

def _get_active_gcp_project() -> str:
    """
    Returns the GCP project ID using the same ADC path you use for BigQuery:
    1) Prefer bigquery.Client().project (matches your schema/table access)
    2) Fallback to google.auth.default()
    """
    try:
        return bigquery.Client().project
    except Exception:
        _, project_id = google.auth.default()
        if not project_id:
            raise RuntimeError("Could not determine active GCP project from ADC.")
        return project_id


@st.cache_resource(show_spinner="Initializing ChromaDB...")
def get_chroma_client():
    """Initialize and return a ChromaDB client with local persistence."""
    try:
        base_dir = Path(__file__).resolve().parent
    except Exception:
        base_dir = Path.cwd()
    persist_dir = base_dir / "chroma_db"
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir),settings=
        Settings(persist_directory=str(persist_dir), anonymized_telemetry=False)
    )


@st.cache_resource
def get_context_collection():
    """Get or create the collection for storing context."""
    client = get_chroma_client()
    return client.get_or_create_collection(name="schema_context")


def get_embedding(text: str) -> list:
    """Get embedding vector for a given text using Vertex AI."""
    model = get_embedding_model()
    embeddings = model.get_embeddings([text])
    return embeddings[0].values


def save_context(table_id: str, column_name: str, description: str, project: str | None = None, dataset: str | None = None):
    """Save or update context in ChromaDB with embeddings.

    Accepts optional `project` and `dataset` to store fully qualified metadata.
    
    NOTE: This function no longer calls client.persist() itself.
    The calling function (e.g., persist_schema_to_chroma) is responsible
    for persisting changes in batch.
    """
    collection = get_context_collection()
    # Create a unique ID for the context
    context_id = f"{table_id}_{column_name}"
    
    # Generate embedding for the description
    embedding = get_embedding(description)
    
    # Prepare the combined text for document (for better semantic search)
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

    # Check if context already exists
    existing = collection.get(
        ids=[context_id],
        include=['metadatas', 'documents', 'embeddings']
    )

    if existing['ids']:
        # Update existing context
        collection.update(
            ids=[context_id],
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    else:
        # Add new context
        collection.add(
            ids=[context_id],
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    # --- REMOVED client.persist() ---
    # We will call persist() once at the end of the batch operation.


def persist_schema_to_chroma(project: str, dataset: str, table: str, schema_df: pd.DataFrame, table_description: str | None = None):
    """Persist a full schema (table + columns) into ChromaDB.

    For each column in `schema_df` this will call `save_context` to add/update
    the column-level context. The function keeps using `table` as the
    `table_id` for compatibility with the existing loading logic.
    """
    # Keep backward compatibility with existing code that uses bare table name
    table_id = table
    for _, row in schema_df.iterrows():
        col = row.get('column_name')
        # Prefer explicitly provided column description, falling back to table-level
        desc = row.get('column_description') if row.get('column_description') is not None else ''
        final_desc = str(desc) if str(desc).strip() else (table_description or '')
        if not col:
            continue
        try:
            save_context(table_id=table_id, column_name=col, description=final_desc, project=project, dataset=dataset)
        except Exception as e:
            # Don't raise—log and continue to avoid breaking the UI flow
            st.warning(f"Failed to save context for {table_id}.{col}: {e}")
            
    # --- ADDED: Persist once after all columns are saved ---
    # try:
    #     client = get_chroma_client()
    #     client.persist()
    # except Exception as e:
    #     st.warning(f"Failed to persist ChromaDB changes to disk: {e}")


def get_context(table_id: str = None, column_name: str = None):
    """Retrieve context from ChromaDB.
    If table_id is None, returns all contexts.
    If column_name is None, returns all contexts for the given table.
    """
    collection = get_context_collection()
    
    if table_id is None:
        # Get all contexts
        return collection.get(include=['metadatas', 'documents', 'embeddings'])
    
    if column_name is None:
        # Get all contexts for a specific table
        return collection.get(
            where={"table_id": table_id},
            include=['metadatas', 'documents', 'embeddings']
        )
    
    # Get specific context
    context_id = f"{table_id}_{column_name}"
    return collection.get(
        ids=[context_id],
        include=['metadatas', 'documents', 'embeddings']
    )


def search_similar_contexts(query: str, n_results: int = 5):
    """Search for similar contexts using semantic similarity."""
    collection = get_context_collection()
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['metadatas', 'documents', 'distances']
    )
    
    return results


def delete_context(table_id: str, column_name: str):
    """Delete a specific context from ChromaDB."""
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="schema_context")
    context_id = f"{table_id}_{column_name}"
    collection.delete(ids=[context_id])
    # try:
    #     client.persist()
    # except Exception as e:
    #     st.warning(f"Failed to persist ChromaDB deletion: {e}")


@st.cache_resource(show_spinner=True)
def get_bq_client():
    """Initializes and returns a BigQuery client."""
    # Uses ADC from `gcloud auth application-default login`
    return bigquery.Client()


# --- REPLACE your get_model() ---
@st.cache_resource(show_spinner="Initializing AI...")
def get_model():
    """Initializes Vertex AI and returns a Gemini model instance."""
    _safe_vertex_init()
    try:
        from vertexai.generative_models import GenerativeModel
        return GenerativeModel("gemini-2.5-flash")
    except Exception:
        # Fallback for older client versions
        from vertexai.generative_models import GenerativeModel
        return GenerativeModel("gemini-1.5-flash")


# --- REPLACE your get_embedding_model() ---
@st.cache_resource(show_spinner="Initializing Embedding Model...")
def get_embedding_model():
    """Initializes and returns a Text Embedding model instance."""
    _safe_vertex_init()
    from vertexai.language_models import TextEmbeddingModel
    candidates = ["gemini-embedding-001","text-embedding-005"]
    last_err = None
    for mid in candidates:
        try:
            m = TextEmbeddingModel.from_pretrained(mid)
            _ = m.get_embeddings(["warmup"])  # fail fast for scopes/API
            return m
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to initialize an embedding model ({', '.join(candidates)}). Last error: {last_err}"
    )


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
        
        # Load descriptions from ChromaDB
        stored_context = get_context(table_id)
        if stored_context['ids']:
            for idx, metadata in enumerate(stored_context['metadatas']):
                column_name = metadata['column_name']
                mask = df['column_name'] == column_name
                if any(mask):
                    # --- MODIFIED: Load from metadata 'description' not 'document' ---
                    # The 'document' has extra text. The 'description' is the raw value.
                    df.loc[mask, 'column_description'] = metadata.get('description', default_desc)
        
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
    # --- MODIFIED: Added 'view_context' to page views ---
    ss.setdefault("view", "analysis") # analysis, context, view_context
    ss.setdefault("auth", False)
    ss.setdefault("ctx_set", False)
    ss.setdefault("selected_tables", [])
    ss.setdefault("schema_df", pd.DataFrame())
    
    # --- NEW: State for BQ pickers ---
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
    ss.setdefault("context_source", None) # 'upload' or 'bq'

_init_state()


# ====================================================================
# --- NEW: CHROMA-BASED PERSISTENCE FUNCTIONS FOR APP STATE ---
# ====================================================================

def persist_current_context_marker(project: str, dataset: str, table: str, description: str | None = None):
    """Persist a small marker in ChromaDB indicating the currently selected context.

    This lets the app reload the last-used context across browser refreshes.
    """
    try:
        client = get_chroma_client()
        coll = client.get_or_create_collection(name="app_state")
        ctx_id = "current_context"
        metadata = {
            "project": project, 
            "dataset": dataset, 
            "table": table, 
            "description": description or ""
        }
        doc = f"Current context: {project}.{dataset}.{table}"

        existing = coll.get(ids=[ctx_id], include=['metadatas', 'documents'])
        if existing['ids']:
            coll.update(ids=[ctx_id], documents=[doc], metadatas=[metadata])
        else:
            coll.add(ids=[ctx_id], documents=[doc], metadatas=[metadata])
        
        # Ensure files are flushed to disk
    #     client.persist()
    except Exception as e:
        st.warning(f"Could not persist app state to ChromaDB: {e}")


def load_persisted_context_to_session():
    """Load persisted context from ChromaDB (if any) into `st.session_state`.

    Attempts to fetch the app-level marker and then reconstruct the full schema
    either from BigQuery (preferred) or from Chroma-stored column contexts.
    
    This should only run ONCE at the start of the app.
    """
    # Only run if context is NOT already set (e.g., by user action in this session)
    if st.session_state.ctx_set:
        return

    try:
        client = get_chroma_client()
        coll = client.get_or_create_collection(name="app_state")
        existing = coll.get(ids=["current_context"], include=['metadatas', 'documents'])
        
        if not existing['ids']:
            return # No persisted state found
            
        meta = existing['metadatas'][0]
        project = meta.get('project')
        dataset = meta.get('dataset')
        table = meta.get('table')
        description = meta.get('description', '')

        if not all([project, dataset, table]):
            return # Invalid state saved

        # Try to fetch the full schema from BigQuery (this will also load descriptions from Chroma)
        try:
            schema = get_table_schema(project, dataset, table)
            if not schema.empty:
                st.session_state.selected_tables = [{"project": project, "dataset": dataset, "table": table, "description": description}]
                st.session_state.schema_df = schema
                st.session_state.context_source = 'bq' # Assume it came from BQ
                set_ctx_if_ready()
                # restore pickers
                st.session_state.pick_project = project
                st.session_state.pick_dataset = dataset
                st.session_state.pick_table = table
                st.session_state.auth = True # We must be auth'd if this worked
                print("Loaded context from BQ via persisted marker.")
                return
        except Exception:
            # Ignore and fallback to rebuilding from Chroma
            print("BQ fetch failed, falling back to Chroma-only context.")
            pass

        # Fallback: reconstruct minimal schema from Chroma-stored column contexts
        contexts = get_context(table)
        if contexts['ids']:
            cols = []
            for idx, md in enumerate(contexts.get('metadatas', [])):
                col_name = md.get('column_name')
                desc = md.get('description', '') # Get raw description
                cols.append({
                    'table_catalog': project or '',
                    'table_schema': dataset or '',
                    'table_name': table,
                    'column_name': col_name,
                    'data_type': 'UNKNOWN', # BQ info was lost
                    'column_description': desc
                })
            
            if cols:
                schema_df = pd.DataFrame(cols)
                st.session_state.selected_tables = [{"project": project, "dataset": dataset, "table": table, "description": description}]
                st.session_state.schema_df = schema_df
                st.session_state.context_source = 'chroma_fallback'
                set_ctx_if_ready()
                st.session_state.pick_project = project
                st.session_state.pick_dataset = dataset
                st.session_state.pick_table = table
                st.session_state.auth = True # Assume auth, but BQ might fail
                print("Loaded context from Chroma fallback.")
                
    except Exception as e:
        # Best-effort loader; do not raise to avoid breaking UI
        st.warning(f"Could not load persisted context: {e}")
        return

# --- LOAD PERSISTED CONTEXT ON SCRIPT RUN ---
# This runs *after* _init_state() and *before* any UI logic
load_persisted_context_to_session()

# ====================================================================


# =========================
# Helper Functions
# =========================
def set_ctx_if_ready():
    """Marks context as set if a table and schema are present."""
    st.session_state.ctx_set = bool(st.session_state.selected_tables) and (
        not st.session_state.schema_df.empty
    )

def set_context(project: str, dataset: str, table: str, description: str, schema_df: pd.DataFrame, source: str):
    """Overwrites the main context with new table information."""
    if not all([project, dataset, table]) or schema_df.empty:
        st.error("Cannot set context with incomplete information.")
        return
    st.session_state.selected_tables = [
        {"project": project, "dataset": dataset, "table": table, "description": description}
    ]
    st.session_state.schema_df = schema_df
    st.session_state.context_source = source 
    set_ctx_if_ready()
    
    # Persist the schema & descriptions to ChromaDB so context survives sessions
    try:
        persist_schema_to_chroma(project, dataset, table, schema_df, description)
        
        # --- MODIFIED: Also persist this as the *current* context ---
        persist_current_context_marker(project, dataset, table, description)
        
    except Exception as e:
        st.warning(f"Failed to persist schema to ChromaDB: {e}")

    st.toast(f"Context set to `{project}.{dataset}.{table}`", icon="🧠")

def clear_context():
    """Clears all context and chat history from the session."""
    st.session_state.selected_tables = []
    st.session_state.schema_df = pd.DataFrame()
    st.session_state.ctx_set = False
    st.session_state.context_source = None 
    st.session_state.messages = []
    
    # --- MODIFIED: Clear the persisted marker ---
    try:
        client = get_chroma_client()
        coll = client.get_or_create_collection(name="app_state")
        coll.delete(ids=["current_context"])
        # client.persist()
    except Exception as e:
        st.warning(f"Could not clear persisted app state: {e}")
        
    st.toast("Context cleared.", icon="🗑️")
    st.rerun()

def add_bq_table_to_context():
    """Sets the selected BQ table as the main context."""
    set_context(
        project=st.session_state.pick_project, 
        dataset=st.session_state.pick_dataset, 
        table=st.session_state.pick_table, 
        description=st.session_state.bq_table_desc,
        schema_df=st.session_state.bq_schema_df,
        source="bq" 
    )
    # Don't clear bq_schema_df, it's needed for the editor view
    # st.session_state.bq_schema_df = pd.DataFrame()
    # st.session_state.bq_table_desc = ""
    st.rerun()

def on_project_change():
    """Resets selections when the GCP project changes."""
    # This callback clears downstream pickers and the schema dataframe
    st.session_state.pick_dataset = None 
    st.session_state.pick_table = None 
    st.session_state.bq_schema_df = pd.DataFrame()
    st.session_state.bq_table_desc = ""

def on_dataset_change():
    """Resets selections when the dataset changes."""
    # This callback clears the table picker and the schema dataframe
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
        # --- NOTE: Assuming 'data/sample_schema.csv' exists ---
        # --- This will fail if the file is not present ---
        # --- Let's create a fallback DataFrame if file not found ---
        
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
        # Reset the description in the callback to avoid widget state errors
        st.session_state.smpl_tbl_desc = ""
    except Exception as e:
        st.error(f"Failed to load sample schema: {e}")
        st.session_state.upload_schema_df = pd.DataFrame()

# --- Gemini Helper Functions ---
def enhance_table_description_llm():
    """Uses the LLM to generate a table description based on its name and schema."""
    model = get_model()
    project = st.session_state.pick_project 
    dataset = st.session_state.pick_dataset 
    table = st.session_state.pick_table 
    schema_df = st.session_state.bq_schema_df

    if not all([project, dataset, table]) or schema_df.empty:
        st.warning("Please select a valid table first.")
        return

    schema_str = "\n".join([f"- {row.column_name} ({row.data_type})" for _, row in schema_df.iterrows()])
    prompt = f"""
    Based on the fully qualified table name `{project}.{dataset}.{table}` and its schema, please provide a concise, one-sentence description of what this table likely contains.
    Consider this table description if given: {st.session_state.bq_table_desc}
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
    project = st.session_state.pick_project 
    dataset = st.session_state.pick_dataset 
    table = st.session_state.pick_table 
    schema_df = st.session_state.bq_schema_df.copy()

    if not all([project, dataset, table]) or schema_df.empty:
        st.warning("Please select a valid table first.")
        return

    client = get_bq_client()

    # Collect lightweight summaries for each column to inform the LLM and to append to descriptions
    summaries = {}
    with st.spinner("Analyzing column statistics in BigQuery..."):
        for _, row in schema_df.iterrows():
            col = row['column_name']
            dtype = (row.get('data_type') or '').upper()
            col_back = f"`{col}`"
            table_fq = f"`{project}.{dataset}.{table}`"

            # Build a small targeted query depending on data type
            if any(t in dtype for t in ['STRING', 'BYTES', 'CHAR']):
                qry = f"SELECT ARRAY_AGG(DISTINCT {col_back} ORDER BY {col_back} LIMIT 10) AS distinct_vals, COUNT(DISTINCT {col_back}) AS distinct_count FROM {table_fq} WHERE {col_back} IS NOT NULL"
            elif any(t in dtype for t in ['DATE', 'TIMESTAMP', 'DATETIME']):
                qry = f"SELECT MIN({col_back}) AS min_val, MAX({col_back}) AS max_val FROM {table_fq} WHERE {col_back} IS NOT NULL"
            elif any(t in dtype for t in ['INT', 'INTEGER', 'NUMERIC', 'FLOAT', 'DOUBLE', 'DECIMAL']):
                # Use APPROX_QUANTILES to avoid scanning huge tables in some cases
                qry = f"SELECT MIN({col_back}) AS min_val, MAX({col_back}) AS max_val, AVG({col_back}) AS avg_val FROM {table_fq} WHERE {col_back} IS NOT NULL"
            elif any(t in dtype for t in ['BOOL', 'BOOLEAN']):
                qry = f"SELECT COUNTIF({col_back}) AS true_count, COUNT(*) - COUNTIF({col_back}) AS false_count, COUNT(*) AS total_count FROM {table_fq}"
            else:
                # Fallback: sample distinct values
                qry = f"SELECT ARRAY_AGG(DISTINCT {col_back} ORDER BY {col_back} LIMIT 10) AS distinct_vals, COUNT(DISTINCT {col_back}) AS distinct_count FROM {table_fq} WHERE {col_back} IS NOT NULL"

            try:
                df_sum = client.query(qry).to_dataframe()
                if not df_sum.empty:
                    # Convert numpy types to python native with json-safe conversions later
                    summaries[col] = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in df_sum.iloc[0].to_dict().items()}
                else:
                    summaries[col] = {}
            except Exception as e:
                summaries[col] = {"error": str(e)}

    # Build prompt including summaries to give the model context about column values
    cols_to_describe = schema_df[['column_name', 'data_type']].to_dict('records')
    # JSON-friendly summaries string
    try:
        summaries_str = json.dumps(summaries, default=str)
    except Exception:
        summaries_str = str(summaries)

    prompt = f"""
Given the table name `{project}.{dataset}.{table}`, and the following per-column lightweight data summaries, provide a concise, one-line description for each of the columns.
Consider this table description if given: {st.session_state.bq_table_desc}
Columns metadata:
{cols_to_describe}

Column value summaries (samples / stats):
{summaries_str}

Please return the output as a simple JSON object where keys are the column names and values are the descriptions. Do not include any other text or markdown formatting.
"""

    with st.spinner("🪄 Enhancing column descriptions..."):
        try:
            response = model.generate_content(prompt)
            json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                desc_dict = json.loads(json_str)
            except json.JSONDecodeError:
                # If the LLM response isn't strict JSON, fall back to a safe heuristic: create descriptions from summaries
                st.warning("AI response was not valid JSON, falling back to heuristic summary.")
                desc_dict = {}

            # Build human-readable appendices from the computed summaries
            def humanize_summary(col_name, summ):
                if not summ:
                    return ""
                if 'error' in summ:
                    return f" (Could not compute data summary)" # Don't show full error
                # Strings / categorical
                if 'distinct_vals' in summ or 'distinct_count' in summ:
                    vals = summ.get('distinct_vals')
                    cnt = summ.get('distinct_count')
                    sample = ', '.join([str(x) for x in (vals or [])]) if vals else ''
                    if cnt is not None:
                        if sample:
                            return f" (Sample distinct values: {sample}; distinct count ≈ {cnt})"
                        return f" (Distinct count ≈ {cnt})"
                # Dates
                if 'min_val' in summ and 'max_val' in summ:
                    mn = summ.get('min_val')
                    mx = summ.get('max_val')
                    if mn is not None and mx is not None:
                        return f" (Value range: {mn} → {mx})"
                # Numeric
                if 'avg_val' in summ or ('min_val' in summ and 'max_val' in summ):
                    mn = summ.get('min_val')
                    mx = summ.get('max_val')
                    avg = summ.get('avg_val')
                    pieces = []
                    if mn is not None and mx is not None:
                        pieces.append(f"range {mn}–{mx}")
                    if avg is not None:
                        try:
                            pieces.append(f"avg {float(avg):.2f}")
                        except Exception:
                            pieces.append(f"avg {avg}")
                    return f" ({'; '.join(pieces)})" if pieces else ""
                # Boolean
                if 'true_count' in summ and 'false_count' in summ:
                    t = summ.get('true_count', 0)
                    fct = summ.get('false_count', 0)
                    tot = summ.get('total_count', t + fct)
                    return f" (True: {t}, False: {fct}, total: {tot})"
                return ''

            # If LLM returned JSON, use that, else fall back to a basic autogenerated description
            final_descriptions = {}
            for _, row in schema_df.iterrows():
                col = row['column_name']
                base = ''
                if isinstance(desc_dict, dict) and col in desc_dict:
                    base = str(desc_dict[col]).strip()
                else:
                    # Fallback heuristics when no LLM description
                    dt = (row.get('data_type') or '').upper()
                    if any(t in dt for t in ['INT','NUMERIC','FLOAT','DOUBLE','DECIMAL']):
                        base = f"Numeric field."
                    elif any(t in dt for t in ['DATE','TIMESTAMP','DATETIME']):
                        base = f"Temporal field."
                    elif any(t in dt for t in ['BOOL','BOOLEAN']):
                        base = f"Boolean flag."
                    else:
                        base = f"Categorical/text field."

                appendix = humanize_summary(col, summaries.get(col, {}))
                final_descriptions[col] = (base + appendix).strip()

            # Apply to schema and update session
            descriptions = pd.Series(final_descriptions)
            schema_df['column_description'] = schema_df['column_name'].map(descriptions).fillna(schema_df['column_description'])
            st.session_state.bq_schema_df = schema_df
            st.rerun()
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
2. Clearly state the goal of the analysis in one concise sentence before listing the steps.
3. Generate a new, concise, step-by-step plan. If a previous plan exists, refine it based on the user's latest feedback.
4. The plan should be clear, unambiguous and easy to understand.
5. If the question cannot be answered using the schema, the plan must explicitly state why.
6. Output only the goal statement and the plan, with the goal as a sentence followed by a numbered or bulleted list. Do not include any preamble, titles, or markdown formatting.
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
        st.error(f"BigQuery Error: {str(e)}")
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


def summarize_query_result_llm(result_df: pd.DataFrame, user_question: str | None = None) -> str:
    """Summarize the given DataFrame using the Gemini model (one key insight in plain English).

    The summarizer now considers the original user question (if provided) so the insight
    is relevant to the user's intent rather than a random observation.

    If the LLM call fails or Vertex AI is not available, fall back to a simple heuristic summary
    that also references the user's question when possible.
    """
    if result_df is None or result_df.empty:
        return "No data available to summarize."

    # Prepare a compact preview to send to the model (limit rows & columns)
    try:
        preview = result_df.head(20).copy()
        # show dtypes and a small sample
        schema_lines = []
        for col in preview.columns:
            schema_lines.append(f"- {col}: {str(preview[col].dtype)}")
        preview_str = preview.to_csv(index=False)

        # Include the user's question in the prompt when available so the AI focuses on that intent
        user_q_section = f"User's question: {user_question}\n\n" if user_question else ""

        prompt = f"""
You are a data analyst. Given a preview of a query result (up to 20 rows), the column types, and the user's question (if provided), provide ONE key insight that best answers or relates to the user's intent.

Include a one-sentence summary that states the most important/high-level insight. Keep it short and do not include any code or extra formatting.

{user_q_section}Column types:
{chr(10).join(schema_lines)}

Data preview (CSV):
{preview_str}

Insight:
"""

        model = get_model()
        with st.spinner("Generating summary from AI..."):
            try:
                response = model.generate_content(prompt)
                summary = response.text.strip()
                # Keep it compact
                return summary
            except Exception as e:
                # Fall through to heuristic fallback
                st.warning(f"AI summarization failed, using local fallback: {e}")
    except Exception as e:
        # If preparing the prompt failed, fall back
        st.warning(f"Failed to prepare data for summarization: {e}")

    # Heuristic fallback: simple statistics-based insight
    try:
        numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            # pick column with largest std dev as potentially most interesting
            stds = result_df[numeric_cols].std(numeric_only=True)
            col = stds.idxmax()
            mean = result_df[col].mean()
            minimum = result_df[col].min()
            maximum = result_df[col].max()
            base_summary = (
                f"The numeric column '{col}' shows the largest variation (std={stds[col]:.2f}) among numeric fields. "
                f"Its values range from {minimum:.2f} to {maximum:.2f} with an average of {mean:.2f}."
            )
            if user_question:
                return f"Key insight related to your question ('{user_question}'): {base_summary}"
            return f"Key insight: {base_summary}"

        # If no numeric columns, look for high-cardinality categorical column or most frequent value
        else:
            cols = result_df.columns.tolist()
            # choose first column and give a frequency-based insight
            col = cols[0]
            top = result_df[col].mode()
            if not top.empty:
                top_val = top.iloc[0]
                freq = (result_df[col] == top_val).sum()
                base_summary = f"In column '{col}', the most common value is '{top_val}' (appears {freq} times in the preview)."
                if user_question:
                    return f"Key insight related to your question ('{user_question}'): {base_summary}"
                return f"Key insight: {base_summary}"
    except Exception as e:
        return f"Could not generate a summary due to an error: {e}"
    
    return "Query executed. No summary could be generated."


# =========================
# Context Page UI
# =========================
def render_ctx_page():
    st.header("Set Context")
    st.caption("Set your context by uploading a schema file or picking a BigQuery table.")

    # --- MODIFIED: Pre-populate edit view if changing context ---
    # On page load, if context is set and upload schema is empty, populate them
    # This makes "Change Context" show the editable view
    if (st.session_state.ctx_set and 
        st.session_state.selected_tables): 
        
        table_info = st.session_state.selected_tables[0]
        current_schema_df = st.session_state.schema_df.copy()
        current_description = table_info.get('description', '')
        context_source = st.session_state.get('context_source')
        
        # Only populate the *specific* tab's state if it's currently empty
        # This check prevents overwriting data if user switches tabs
        if context_source == "upload" and st.session_state.upload_schema_df.empty:
            st.session_state.upload_schema_df = current_schema_df
            st.session_state.smpl_tbl_desc = current_description
            
        elif context_source == "bq" and st.session_state.bq_schema_df.empty:
            st.session_state.bq_schema_df = current_schema_df
            st.session_state.bq_table_desc = current_description
            # Only set pickers if they are not already set
            if st.session_state.pick_project is None:
                st.session_state.pick_project = table_info['project']
            if st.session_state.pick_dataset is None:
                st.session_state.pick_dataset = table_info['dataset']
            if st.session_state.pick_table is None:
                st.session_state.pick_table = table_info['table']
            st.session_state.auth = True
    # --- END MODIFIED LOGIC ---

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
            # Handle potential missing columns if CSV is malformed
            try:
                prj = df_editor['table_catalog'].unique()[0]
                dtset = df_editor['table_schema'].unique()[0]
                tbl = df_editor['table_name'].unique()[0]
            except (KeyError, IndexError):
                st.error("Uploaded CSV is missing required columns: 'table_catalog', 'table_schema', 'table_name'.")
                st.session_state.upload_schema_df = pd.DataFrame() # Clear the bad df
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

            c3, c4 = st.columns(2)
            if c3.button("Enhance Schema", icon="🪄", help="A.I. will populate the column description for you", key="enhance_upload_schema"):
                # --- NOTE: This assumes 'data/sample_schema_with_desc.csv' exists ---
                try:
                    # --- Let's create a fallback DataFrame if file not found ---
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
            if c4.button("Set as Context", icon="🧠", key="set_context_upload_btn"):
                set_context(prj, dtset, tbl, st.session_state.smpl_tbl_desc, edited_schema, source="upload") # <--- MODIFIED
                st.session_state.upload_schema_df = pd.DataFrame()
                # st.session_state.smpl_tbl_desc = ""
                st.rerun()

    with tab_pick:
        st.subheader("1. Authenticate with Google")
        c1, c2 = st.columns([2, 5], vertical_alignment="center", gap="small")
        if c1.button("Login to Google", icon="🔑", use_container_width=True, disabled=st.session_state.auth):
            st.session_state.auth = True
            st.rerun() # Rerun to show success
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

            # --- MODIFIED: Use key to manage state, no redundant assignment ---
            project = st.selectbox("GCP Project", [""] + projects, placeholder="Select a project", key="pick_project", on_change=on_project_change)

            project_val = st.session_state.pick_project
            datasets = list_datasets(project_val) if project_val else []
            dataset = st.selectbox("Dataset", [""] + datasets, placeholder="Select a dataset", key="pick_dataset", on_change=on_dataset_change)

            dataset_val = st.session_state.pick_dataset
            tables = list_tables(project_val, dataset_val) if project_val and dataset_val else []
            table = st.selectbox("Table", [""] + tables, placeholder="Select a table", key="pick_table")

            table_val = st.session_state.pick_table
            
            if all([project_val, dataset_val, table_val]): # <-- MODIFIED
                # This block now only runs if the schema is empty or the table changed
                if st.session_state.bq_schema_df.empty or st.session_state.bq_schema_df['table_name'].iloc[0] != table_val: # <-- MODIFIED
                    st.session_state.bq_schema_df = get_table_schema(project_val, dataset_val, table_val) # <-- MODIFIED
                    st.session_state.bq_table_desc = ""
                    st.rerun() # Rerun to populate the editor
                
                c1, c2 = st.columns([9, 2])
                c1.text_area("Table Description", key="bq_table_desc", placeholder="Add custom description for this table", label_visibility="collapsed")
                c2.button("Enhance", icon="🪄", 
                            help="A.I. will populate table description", 
                            on_click=enhance_table_description_llm, 
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
                    c3, c4 = st.columns(2)
                    c3.button("Enhance Schema", icon="🪄", help="A.I. will populate column descriptions", on_click=enhance_column_descriptions_llm, key="enhance_bq_cols")
                    c4.button("Set as Context", icon="🧠", on_click=add_bq_table_to_context, key="set_context_bq_btn")

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
# --- NEW: View Context Page UI ---
# =========================
def render_view_context_page():
    """Displays the currently set context (table and schema)."""
    st.header("View Current Context")

    if not st.session_state.ctx_set or not st.session_state.selected_tables:
        st.error("No context is currently set.")
        if st.button("Set Context", icon="🧠", use_container_width=True):
            st.session_state.view = "context"
            st.rerun()
        return

    # Display current context
    table_info = st.session_state.selected_tables[0]
    schema_df = st.session_state.schema_df

    st.subheader("Table Information")
    st.markdown(f"**Name:** `{table_info['project']}.{table_info['dataset']}.{table_info['table']}`")
    
    st.markdown("**Description:**")
    if table_info.get('description'):
        st.info(table_info['description'])
    else:
        st.caption("No description provided for this table.")

    st.divider()

    st.subheader("Table Schema")
    if schema_df.empty:
        st.warning("Schema data is missing from the current context.")
    else:
        # Use st.dataframe for a read-only view
        st.dataframe(
            schema_df, 
            hide_index=True,
            use_container_width=True,
            column_config={
                "column_description": st.column_config.TextColumn("Column Description")
            }
        )
    
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Return to Analysis", icon="⬅️", use_container_width=True):
            st.session_state.view = "analysis"
            st.rerun()
    with c2:
        if st.button("Change Context", icon="🔄", use_container_width=True):
            st.session_state.view = "context"
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
        st.markdown(
            "<div style='background-color:#fff3cd;border:1px solid #ffeeba;color:#856404;padding:12px;border-radius:4px;text-align:center;'>"
            "<span style='font-size:1.05em;'>⚠️ Context not set. Please set context in the left sidebar to proceed.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # return # Keep rendering the rest of the page

    st.markdown("<h2 style='text-align:center;margin-top:0;'>🧐 What are you analyzing today?</h2>", unsafe_allow_html=True)
    
    if not st.session_state.ctx_set:
        st.chat_input("Set context to start analyzing your data.", disabled=True)
        return
    
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

                    # If a summary was previously generated for this result, display it
                    if "summary" in msg:
                        st.info(msg["summary"])

                    # Summarize button: analyze the full result (or preview) and return one key insight
                    if st.button("📝 Summarize", key=f"summarize_{i}"):
                        full_df = msg.get("full_query_result", msg.get("query_result"))
                        # Attempt to find the original user question that led to this SQL/result
                        original_prompt = ""
                        for j in range(i, -1, -1):
                            if st.session_state.messages[j]["role"] == "user":
                                original_prompt = st.session_state.messages[j]["content"]
                                break
                        summary_text = summarize_query_result_llm(full_df, user_question=original_prompt)
                        st.session_state.messages[i]["summary"] = summary_text
                        # No need for an extra chat message, just rerun to show the st.info
                        st.rerun()
                    
                    if msg.get("chart_created"):
                        st.caption("Chart:")
                        chart_info = msg["chart_suggestion"]
                        chart_df = msg["full_query_result"] # Use full result for charting
                        
                        # Verify columns exist before trying to chart
                        if chart_info["x_axis"] in chart_df.columns and chart_info["y_axis"] in chart_df.columns:
                            # Streamlit charts often work best with the x-axis set as the index
                            try:
                                chart_df_indexed = chart_df.set_index(chart_info["x_axis"])
                            except Exception:
                                # Fallback if index can't be set (e.g., non-unique values)
                                chart_df_indexed = chart_df
                                
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
                    # Support manual editing of the SQL before running
                    # If the message is in edit mode, show the editor + Update/Cancel
                    if msg.get("editable"):
                        # Provide a multiline editor pre-filled with edited_sql or sql_text
                        current_sql = msg.get("edited_sql", msg.get("sql_text", ""))
                        edited = st.text_area("Edit SQL", value=current_sql, key=f"manual_sql_editor_{i}", height=220)
                        c_upd, c_can = st.columns([1, 1])
                        if c_upd.button("Update", key=f"update_sql_{i}"):
                            # Save the edited SQL back to the message and exit edit mode
                            st.session_state.messages[i]["sql_text"] = edited
                            st.session_state.messages[i]["display_content"] = f"**Generated SQL:**\n```sql\n{edited}\n```"
                            # clear edit state
                            st.session_state.messages[i].pop("editable", None)
                            st.session_state.messages[i].pop("edited_sql", None)
                            st.rerun()
                        if c_can.button("Cancel", key=f"cancel_edit_{i}"):
                            # Discard edits and exit edit mode
                            st.session_state.messages[i].pop("editable", None)
                            st.session_state.messages[i].pop("edited_sql", None)
                            st.rerun()
                        # Keep the edited value in session so it persists while typing
                        st.session_state.messages[i]["edited_sql"] = edited
                    else:
                        # Not in edit mode: show Manual Edit and Run buttons side-by-side
                        c_run, c_edit = st.columns([1, 1])
                        if c_edit.button("✏️ Manual Edit", key=f"manual_edit_{i}"):
                            st.session_state.messages[i]["editable"] = True
                            st.session_state.messages[i]["edited_sql"] = msg.get("sql_text", "")
                            st.rerun()

                        if c_run.button("🚀 Run SQL", key=f"run_sql_{i}"):
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
    
    if prompt := st.chat_input(placeholder, disabled=not st.session_state.ctx_set):
        st.session_state.messages.append({"role": "user", "display_content": prompt, "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                conversation_text = build_llm_conversation_text(st.session_state.messages)
                new_plan_text = generate_plan_llm(conversation_text)

                if new_plan_text:
                    display_content = f"**Here is the goal and proposed plan:**\n\n{new_plan_text}"
                    st.session_state.messages.append({
                        "role": "assistant", "type": "plan",
                        "plan_text": new_plan_text, "display_content": display_content,
                        "approved": False
                    })
        st.rerun()


# =========================
# --- MODIFIED: Sidebar UI ---
# =========================
st.sidebar.title("F.R.I.D.A.Y")
st.sidebar.caption("AI-Powered Analytics Assistant")

st.sidebar.title("Context")
if not st.session_state.ctx_set:
    st.sidebar.warning("Context is empty", icon="⚠️")
    if st.sidebar.button("Set Context", icon="🧠", use_container_width=True):
        st.session_state.view = "context"
        st.rerun()
else:
    st.sidebar.success("Context set", icon="✅")
    if st.session_state.selected_tables:
        row = st.session_state.selected_tables[0]
        st.sidebar.write(f"**Table:** `{row['project']}.{row['dataset']}.{row['table']}`")
        if row.get("description"):
            st.sidebar.caption(row['description'])
    if not st.session_state.schema_df.empty:
        st.sidebar.caption(f"Schema: loaded ({st.session_state.get('context_source', 'N/A')})")

    if st.sidebar.button("View Context Details", icon="📄", use_container_width=True):
        st.session_state.view = "view_context"
        st.rerun()
    
    if st.sidebar.button("Change Context", icon="🔄", use_container_width=True):
        st.session_state.view = "context"
        st.rerun()

with st.sidebar.expander("ℹ️ How it works", expanded=True):
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
# --- MODIFIED: Main Router ---
# =========================
if st.session_state.view == "context":
    render_ctx_page()
elif st.session_state.view == "view_context":
    render_view_context_page()
else:
    # Default to analysis view
    st.session_state.view = "analysis"
    render_default_page()