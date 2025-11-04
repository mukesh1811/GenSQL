## auth
# gcloud auth login
# gcloud auth application-default login

## set project
# gcloud config set project aeo-supplychain-datamart-prod
# gcloud auth application-default set-quota-project aeo-supplychain-datamart-prod

## set project
# gcloud config set project learning-prj-id
# gcloud auth application-default set-quota-project learning-prj-id

import streamlit as st
import pandas as pd
import json
import asyncio
import traceback # To show detailed errors

from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError
import google.auth

# --- Google AI SDK Imports ---
from vertexai.language_models import TextGenerationModel
from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Content

# =============================================================================
# 1. CORE CACHED FUNCTIONS (Get BQ Client)
# =============================================================================

@st.cache_resource(show_spinner="Connecting to BigQuery & Vertex AI...")
def get_bq_client_and_init_vertex():
    """Initializes and returns a BigQuery client and inits Vertex AI."""
    try:
        credentials, project_id = google.auth.default()
        if not project_id:
            # Try to get project ID from credentials if not directly available
            project_id = credentials.quota_project_id
        if not project_id:
             # If still no project ID, try getting it from gcloud config
             try:
                 import subprocess
                 project_id = subprocess.check_output(
                     ["gcloud", "config", "get-value", "project"],
                     stderr=subprocess.STDOUT,
                     text=True,
                 ).strip()
             except Exception:
                 pass # Ignore errors if gcloud is not installed or configured

        if not project_id:
             st.error("Could not determine Google Cloud project ID. Please set using `gcloud config set project YOUR_PROJECT_ID` and `gcloud auth application-default login`")
             st.stop()

        # Initialize Vertex AI
        aiplatform.init(
            project=project_id,
            location="us-central1",  # Change this to your desired region
            credentials=credentials
        )

        return bigquery.Client(credentials=credentials, project=project_id)
    except google.auth.exceptions.DefaultCredentialsError as cred_error:
         st.error(f"Authentication Error: Could not find Application Default Credentials. Please run `gcloud auth application-default login`. Details: {cred_error}")
         st.stop()
    except Exception as e:
        st.error(f"Failed to initialize Google Cloud clients: {e}")
        st.code(traceback.format_exc())
        st.stop()

# Call the init function
bq_client = get_bq_client_and_init_vertex()

# =============================================================================
# 2. AGENT TOOLS (The Agent's "Hands")
# =============================================================================

def get_table_schema(project_id: str, dataset_id: str, table_id: str) -> str:
    """
    Fetches the schema (column names and data types) for a specific BigQuery table.

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The BigQuery table ID.

    Returns:
        A JSON string of the schema (list of dicts) or an error object.
    """
    if not bq_client:
        return json.dumps({"error": "BigQuery client not available."})

    query = f"""
        SELECT column_name, data_type
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_id}'
        ORDER BY ordinal_position
    """
    try:
        df = bq_client.query(query).to_dataframe()
        if df.empty:
             return json.dumps({"error": f"Table or schema not found (or table has no columns): {project_id}.{dataset_id}.{table_id}"})
        return df.to_json(orient='records')
    except GoogleAPIError as api_error:
         # Handle specific BQ errors like Not Found
         if "Not found" in str(api_error):
              return json.dumps({"error": f"Table not found: {project_id}.{dataset_id}.{table_id}. Details: {api_error}"})
         else:
              return json.dumps({"error": f"BigQuery API Error fetching schema for {project_id}.{dataset_id}.{table_id}: {api_error}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to get schema for {project_id}.{dataset_id}.{table_id}: {e}"})

def find_similar_tables(project_id: str, dataset_id: str, table_id: str) -> str:
    """
    Finds other tables in the same dataset with similar names (using a simple prefix logic),
    which might cause ambiguity (e.g., 'sales_daily' vs 'sales_weekly').

    Args:
        project_id: The Google Cloud project ID.
        dataset_id: The BigQuery dataset ID.
        table_id: The name of the table to find matches for.

    Returns:
        A JSON string of the list of similar table names (or empty list / error object).
    """
    if not bq_client:
        return json.dumps({"error": "BigQuery client not available."})

    # Simple logic to find a "base name" by removing common suffixes
    base_name = table_id
    for suffix in ['_daily', '_weekly', '_monthly', '_yearly', '_fact', '_dim', '_agg']:
         if base_name.endswith(suffix):
              base_name = base_name[:-len(suffix)]
              break # Only remove one suffix type

    # If the name itself ends with a digit, it might be part of sharding, less likely a prefix
    if base_name[-1].isdigit():
        print(f"Skipping similar table search for potentially sharded table: {table_id}")
        return json.dumps([])


    query = f"""
        SELECT table_name
        FROM `{project_id}.{dataset_id}.INFORMATION_SCHEMA.TABLES`
        WHERE STARTS_WITH(table_name, '{base_name}')
          AND table_name != '{table_id}'
          AND table_type = 'BASE TABLE' /* Exclude views unless specified */
        LIMIT 5 /* Limit results to avoid overwhelming the agent */
    """
    try:
        df = bq_client.query(query).to_dataframe()
        return df.to_json(orient='records') if not df.empty else json.dumps([])
    except Exception as e:
        # Don't treat this as fatal, just log and return empty
        print(f"Warning: Error finding similar tables for {table_id}: {e}")
        return json.dumps([])

def run_profiling_query(sql_query: str) -> str:
    """
    Runs a SQL query against BigQuery and returns the result as a JSON string.
    Use this to find the 'grain' of a table (e.g., COUNT(DISTINCT date_col))
    or to find example values (e.g., SELECT DISTINCT status_col FROM ...).

    Args:
        sql_query: The SQL query to run.

    Returns:
        A JSON string of the query result or an error object.
    """
    if not bq_client:
        return json.dumps({"error": "BigQuery client not available."})

    # Basic check to prevent overly broad queries in demo
    if "LIMIT" not in sql_query.upper() and ("COUNT(" not in sql_query.upper() and "APPROX_COUNT_DISTINCT(" not in sql_query.upper()):
        # Add a default limit if not a count query
        sql_query_limited = f"{sql_query} LIMIT 20"
        print(f"Warning: Adding default LIMIT 20 to profiling query: {sql_query}")
    else:
        sql_query_limited = sql_query

    try:
        df = bq_client.query(sql_query_limited).to_dataframe()

        # Convert datetime-like columns for JSON serialization if needed.
        # Use only pandas-recognized datetime dtype names and provide a fallback
        # that checks each column with pandas' type-checking utility.
        # try:
        #     datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, tz]']).columns
        # except Exception:
        #     from pandas.api.types import is_datetime64_any_dtype
        #     datetime_cols = [c for c in df.columns if is_datetime64_any_dtype(df[c])]
        # for col in datetime_cols:
        #     df[col] = df[col].astype(str)
        
        return df.to_json(orient='records',date_format='iso')
    except Exception as e:
        return json.dumps({"error": f"Query failed for '{sql_query_limited}': {e}"})


# -----------------------
# Column profiling helpers
# -----------------------
def _safe_parse_single_record(json_text: str):
    """Return first record dict or None on error."""
    try:
        data = json.loads(json_text)
        print(data)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return data[0]
    except Exception:
        pass
    return None


def _extract_embedded_json(text: str):
    """Try to extract a JSON object or array embedded in arbitrary text.

    Returns the JSON substring or None if not found.
    This scans for a balanced '{' ... '}' or '[' ... ']' block.
    """
    if not isinstance(text, str):
        return None

    text = text.strip()
    # Quick checks: if the whole text is JSON-like already
    if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
        return text

    # Search for a balanced JSON object
    for start_char, end_char in ('{', '}'), ('[', ']'):
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        depth = 0
        for i in range(start_idx, len(text)):
            ch = text[i]
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx:i+1]
                    return candidate
    return None


def profile_column(project_id: str, dataset_id: str, table_id: str, column_name: str, data_type: str) -> str:
    """Run a small set of profiling queries for a single column and return a concise summary string.

    Uses `run_profiling_query` for all queries and is defensive about errors.
    """
    col = column_name
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    summary_parts = []

    try:
        dt = (data_type or "").upper()
        print(f"[profiler] Starting profiling for {project_id}.{dataset_id}.{table_id}.{col} (type={dt})")

        # DATE / TIMESTAMP types: get min/max
        if 'DATE' in dt or 'TIMESTAMP' in dt or 'TIME' in dt:
            q = f"SELECT MIN(`{col}`) AS min_val, MAX(`{col}`) AS max_val FROM `{table_ref}`"
            res = run_profiling_query(q)
            r = _safe_parse_single_record(res)
            if r:
                summary_parts.append(f"Range: {r.get('min_val')} -> {r.get('max_val')}")

        # NUMERIC types: min/max/avg/stddev
        elif any(x in dt for x in ['INT', 'FLOAT', 'NUMERIC', 'DECIMAL']) and col != 'sku':
            q = f"SELECT MIN(`{col}`) AS min_val, MAX(`{col}`) AS max_val, AVG(`{col}`) AS avg_val, STDDEV_POP(`{col}`) AS stddev FROM `{table_ref}`"
            res = run_profiling_query(q)
            r = _safe_parse_single_record(res)
            if r:
                summary_parts.append(f"Min: {r.get('min_val')}, Max: {r.get('max_val')}, Avg: {r.get('avg_val')}, Std: {r.get('stddev')}")

        # BOOLEAN
        elif 'BOOL' in dt or 'BOOLEAN' in dt:
            q = f"SELECT COUNTIF(`{col}` = TRUE) AS true_count, COUNTIF(`{col}` = FALSE) AS false_count, COUNT(*) AS total_count FROM `{table_ref}`"
            res = run_profiling_query(q)
            r = _safe_parse_single_record(res)
            if r:
                summary_parts.append(f"True: {r.get('true_count')}, False: {r.get('false_count')}, Total: {r.get('total_count')}")

        # Strings / Text / Others: check distinct count first to detect categorical
        else:
            # approximate distinct count
            q_dist = f"SELECT APPROX_COUNT_DISTINCT(`{col}`) AS distinct_count FROM `{table_ref}`"
            resd = run_profiling_query(q_dist)
            rd = _safe_parse_single_record(resd)
            distinct_count = None
            try:
                if rd and 'distinct_count' in rd:
                    distinct_count = int(rd['distinct_count'])
            except Exception:
                distinct_count = None

            if distinct_count is not None and distinct_count <= 10:
                # Fetch distinct values to show examples
                q_vals = f"SELECT DISTINCT `{col}` AS val FROM `{table_ref}` ORDER BY val LIMIT 20"
                resv = run_profiling_query(q_vals)
                try:
                    vals = json.loads(resv)
                    examples = [str(r.get('val')) for r in vals if isinstance(r, dict) and 'val' in r]
                    summary_parts.append(f"Categorical (~{distinct_count} distinct). Examples: {', '.join(examples[:10])}")
                except Exception:
                    summary_parts.append(f"Categorical (~{distinct_count} distinct values)")
            else:
                # Provide sample values and average length (for strings)
                q_sample = f"SELECT `{col}` AS sample_val FROM `{table_ref}` WHERE `{col}` IS NOT NULL LIMIT 10"
                resample = run_profiling_query(q_sample)
                try:
                    sdata = json.loads(resample)
                    samples = [str(r.get('sample_val')) for r in sdata if isinstance(r, dict) and 'sample_val' in r]
                    if samples:
                        summary_parts.append(f"({distinct_count} distinct values). Sample values: {', '.join(samples[:5])}")
                except Exception:
                    pass

                # If string-like, try avg length
                if 'CHAR' in dt or 'STRING' in dt or 'TEXT' in dt:
                    q_len = f"SELECT AVG(LENGTH(`{col}`)) AS avg_len FROM `{table_ref}` WHERE `{col}` IS NOT NULL"
                    rlen = run_profiling_query(q_len)
                    rl = _safe_parse_single_record(rlen)
                    if rl and rl.get('avg_len') is not None:
                        summary_parts.append(f"Avg length: {rl.get('avg_len')}")

    except Exception as e:
        # Non-fatal: return what we have plus an error note
        summary_parts.append(f"(profiling error: {e})")
        print(f"[profiler] Error profiling {col}: {e}")

    summary = ' | '.join(summary_parts)
    if summary:
        print(f"[profiler] Summary for {col}: {summary}")
    else:
        print(f"[profiler] No profiling summary generated for {col}")
    return summary

# =============================================================================
# 3. AGENT DEFINITION (The "Brain")
# =============================================================================

@st.cache_resource(show_spinner="Warming up the agent... 🧠")
def get_gemini_model():
    """Creates and caches the Gemini Model."""
    
    tools = [
        {
            "name": "get_table_schema",
            "description": "Fetches the schema for a specific BigQuery table",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "The Google Cloud project ID"},
                    "dataset_id": {"type": "string", "description": "The BigQuery dataset ID"},
                    "table_id": {"type": "string", "description": "The BigQuery table ID"}
                },
                "required": ["project_id", "dataset_id", "table_id"]
            }
        },
        {
            "name": "find_similar_tables",
            "description": "Finds other tables with similar names in the same dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "The Google Cloud project ID"},
                    "dataset_id": {"type": "string", "description": "The BigQuery dataset ID"},
                    "table_id": {"type": "string", "description": "The name of the table to find matches for"}
                },
                "required": ["project_id", "dataset_id", "table_id"]
            }
        },
        {
            "name": "run_profiling_query",
            "description": "Runs a SQL query against BigQuery",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string", "description": "The SQL query to run"}
                },
                "required": ["sql_query"]
            }
        }
    ]

    # Create the model with function calling capabilities
    model = GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config={
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            # "max_output_tokens": 2048,
        }
    )
    
    return model

# --- Initialize the Model ---
try:
    model = get_gemini_model()
except Exception as e:
    st.error(f"Failed to create Gemini model: {e}")
    st.code(traceback.format_exc())
    st.stop()

# =============================================================================
# 4. STREAMLIT UI
# =============================================================================

st.set_page_config(page_title="Auto-Curator Agent (ADK)", layout="wide")
st.title("Auto-Curator Agent Demo 🤖 (powered by Google ADK)")
st.caption("This app uses an AI agent to automatically profile a BigQuery table and generate high-quality, 'semantically-aware' descriptions.")

if "table_desc" not in st.session_state: st.session_state.table_desc = ""
if "column_desc_df" not in st.session_state: st.session_state.column_desc_df = pd.DataFrame(columns=["column_name", "data_type", "description", "profile_summary"])

st.header("1. Select a Table to Profile")
st.info("Ensure your Application Default Credentials (`gcloud auth application-default login`) have access to the selected GCP Project.", icon="🔑")

# Use public data for default example
# DEFAULT_PROJECT = "aeo-datasci-common-prod"
# DEFAULT_DATASET = "Lost_sales"
# DEFAULT_TABLE = "LOST_SALES_COMMON"

# DEFAULT_PROJECT = "bigquery-public-data"
# DEFAULT_DATASET = "samples"
# DEFAULT_TABLE = "wikipedia"

# DEFAULT_PROJECT = "learning-prj-id"
# DEFAULT_DATASET = "forecasting_sticker_sales"
# DEFAULT_TABLE = "train"

DEFAULT_PROJECT = "aeo-supplychain-datamart-prod"
DEFAULT_DATASET = "runnelsg"
DEFAULT_TABLE = "ls_with_boss"

col1, col2, col3 = st.columns(3)
project_id = col1.text_input("GCP Project ID", DEFAULT_PROJECT)
dataset_id = col2.text_input("Dataset ID", DEFAULT_DATASET)
table_id = col3.text_input("Table ID", DEFAULT_TABLE)

st.markdown(f"**Target Table:** `{project_id}.{dataset_id}.{table_id}`")

# --- Helper function to run Gemini in Streamlit ---
def run_gemini(status_box, user_input_text: str):
    """Runs Gemini and updates the Streamlit status box."""
    
    # Create the chat history with system prompt and user message
    system_prompt = """You are an expert 'Auto-Curator' data analyst. Your goal is to generate a high-quality, semantically rich description for a given BigQuery table, specified by the user as project.dataset.table.

Your task is to analyze the table and create a JSON response with:
1. table_description: A concise purpose and grain summary
2. column_descriptions: Context-aware descriptions for EVERY column in the schema

CRITICAL REQUIREMENTS:
- You MUST provide descriptions for ALL columns in the schema, no exceptions
- Each column description should explain:
  * The purpose and meaning of the column
  * Business context and relationships with other columns
  * Any relevant data quality notes (e.g., nullable, constraints)
- The column_descriptions object MUST contain an entry for every column from the schema

To accomplish this:
1. First get the schema using the SQL: SELECT column_name, data_type FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS` WHERE table_name = '{table}'
2. Look for date/timestamp columns to understand the grain (daily, weekly, monthly)
3. Use COUNT(DISTINCT date_col) to determine granularity
4. Create descriptions that reflect the table's purpose and grain. Include a single line explaining the available columns in this table and what kind of data is this.
5. Verify that every column from step 1 has a description before returning

Return ONLY a valid JSON object with:
{
    "table_description": "overall description",
    "column_descriptions": {
        "column1": "detailed description",
        "column2": "detailed description",
        ...ALL columns must be included
    }
}"""

    try:
        chat = model.start_chat(history=[])
        full_prompt = f"{system_prompt}\n\nUser request: {user_input_text}"
        
        # Send message without streaming
        response = chat.send_message(
            Content(
                parts=[Part.from_text(full_prompt)],
                role="user"
            )
        )
        
        if response and response.text:
            status_box.write("✅ Done!")
            return response.text
        
        return "Error: No response generated"

    except Exception as e:
        status_box.error(f"Error during Gemini execution: {str(e)}")
        st.exception(e)
        return None

    except Exception as e:
        st.error(f"An error occurred during agent execution: {e}")
        st.code(traceback.format_exc())
        return None

# --- Button to run the agent ---
if st.button(f"✨ Auto-Profile Table"):

    # Basic input validation
    if not project_id or not dataset_id or not table_id:
        st.warning("Please provide Project ID, Dataset ID, and Table ID.")
    else:
        st.session_state.table_desc = ""
        st.session_state.column_desc_df = pd.DataFrame(columns=["column_name", "data_type", "description", "profile_summary"])

        user_input_text = f"Please profile this table: {project_id}.{dataset_id}.{table_id}"

        with st.status("🤖 **Auto-Curator Agent is working...**", expanded=True) as status:
            final_output_text = run_gemini(status, user_input_text)

            if final_output_text:
                try:
                    # Clean up markdown backticks and try to robustly extract embedded JSON
                    raw_text = final_output_text or ""
                    clean_json_text = raw_text.strip()

                    # If wrapped in triple-backticks with json, unwrap those first
                    if clean_json_text.startswith("```json") and clean_json_text.endswith("```"):
                        clean_json_text = clean_json_text[7:-3].strip()
                    elif clean_json_text.startswith("```") and clean_json_text.endswith("```"):
                        clean_json_text = clean_json_text[3:-3].strip()

                    # Attempt to extract a JSON object/array embedded anywhere in the text
                    extracted = _extract_embedded_json(clean_json_text)
                    if extracted:
                        clean_json_text = extracted

                    final_answer_json = json.loads(clean_json_text)

                    # Validate the structure of the JSON
                    if not isinstance(final_answer_json, dict) or \
                       "table_description" not in final_answer_json or \
                       "column_descriptions" not in final_answer_json:
                        raise ValueError("Agent JSON output missing required keys ('table_description', 'column_descriptions')")

                    st.session_state.table_desc = final_answer_json.get("table_description")

                    # If table description indicates an error, show it prominently
                    if isinstance(st.session_state.table_desc, str) and st.session_state.table_desc.lower().startswith("error"):
                         st.warning(f"Agent reported an error: {st.session_state.table_desc}")

                    # Re-hydrate the schema DataFrame with new descriptions
                    base_schema_json = get_table_schema(project_id, dataset_id, table_id)
                    base_schema = []
                    schema_error = None
                    try:
                        schema_data = json.loads(base_schema_json)
                        if isinstance(schema_data, dict) and "error" in schema_data:
                            schema_error = schema_data['error']
                        elif isinstance(schema_data, list):
                            base_schema = schema_data
                        else:
                            schema_error = f"Unexpected schema format: {type(schema_data)}"
                    except json.JSONDecodeError:
                        schema_error = f"Could not parse base schema JSON: {base_schema_json}"

                    if schema_error:
                        st.warning(f"Could not fetch or parse base schema: {schema_error}")
                        # Still try to show table description if agent provided one
                        st.session_state.column_desc_df = pd.DataFrame(columns=["column_name", "data_type", "description", "profile_summary"]) # Ensure empty dataframe with expected columns

                    col_desc_dict = final_answer_json.get("column_descriptions", {})
                    if not isinstance(col_desc_dict, dict):
                         st.warning("Agent returned invalid format for 'column_descriptions' (expected a JSON object).")
                         col_desc_dict = {}

                    if base_schema:
                        processed_rows = []
                        missing_cols_from_agent = []
                        schema_cols = {row.get('column_name') for row in base_schema if isinstance(row, dict)}
                        
                        # First, identify any missing columns
                        for row in base_schema:
                            if isinstance(row, dict) and 'column_name' in row:
                                col_name = row['column_name']
                                if col_name not in col_desc_dict:
                                    missing_cols_from_agent.append(col_name)

                        # If we have missing columns, try to get descriptions for them specifically
                        if missing_cols_from_agent:
                            retry_prompt = f"""Please provide descriptions ONLY for these specific columns that were missed in the previous response: {', '.join(missing_cols_from_agent)}
                            
Return a JSON object with ONLY the missing column descriptions in this format:
{{
    "column_descriptions": {{
        "missed_column1": "description",
        "missed_column2": "description"
    }}
}}"""
                            
                            retry_response = model.start_chat().send_message(
                                Content(
                                    parts=[Part.from_text(retry_prompt)],
                                    role="user"
                                )
                            )
                            
                            if retry_response and retry_response.text:
                                try:
                                    # Clean up markdown backticks if present
                                    clean_retry_json = retry_response.text.strip()
                                    if clean_retry_json.startswith("```json"):
                                        clean_retry_json = clean_retry_json[7:-4].strip()
                                    elif clean_retry_json.startswith("```"):
                                        clean_retry_json = clean_retry_json[3:-3].strip()
                                    
                                    retry_data = json.loads(clean_retry_json)
                                    if isinstance(retry_data, dict) and "column_descriptions" in retry_data:
                                        # Update the main dictionary with retry results
                                        col_desc_dict.update(retry_data["column_descriptions"])
                                except Exception as e:
                                    st.warning(f"Failed to get descriptions for missing columns: {e}")

                        # Now process all rows with any additional descriptions we got
                        for row in base_schema:
                            if isinstance(row, dict) and 'column_name' in row:
                                col_name = row['column_name']
                                description = col_desc_dict.get(col_name)
                                if description:
                                    row['description'] = description
                                else:
                                    row['description'] = "<description not generated by agent>"
                                    # Keep track of truly missing descriptions after retry
                                    if col_name in missing_cols_from_agent:
                                        st.warning(f"Failed to generate description for column: {col_name}")
                                # Run lightweight profiling to enrich the description
                                try:
                                    prof = profile_column(project_id, dataset_id, table_id, col_name, row.get('data_type'))
                                    print(f"[profiler] Called profile_column for {col_name}, got: {prof}")
                                    # Store profiling summary in its own column so UI can render it reliably
                                    row['profile_summary'] = prof or ''
                                except Exception as e:
                                    # Don't fail the whole flow for profiling errors
                                    print(f"Warning: profiling failed for {col_name}: {e}")
                                    row['profile_summary'] = f"(profiling error: {e})"

                                processed_rows.append(row)
                            else:
                                st.warning(f"Skipping invalid row in fetched schema: {row}")

                        st.session_state.column_desc_df = pd.DataFrame(processed_rows)

                        # Check if agent provided descriptions for columns not in the actual schema
                        agent_extra_cols = set(col_desc_dict.keys()) - schema_cols
                        if agent_extra_cols:
                             st.warning(f"Agent provided descriptions for columns not found in the actual schema: {', '.join(agent_extra_cols)}")

                    status.update(label="Profiling Complete!", state="complete", expanded=False)

                except json.JSONDecodeError:
                    status.write("Error: Agent returned invalid JSON. Could not parse the final output. See raw output below:")
                    st.code(final_output_text, language="text")
                    status.update(label="Agent Error: Invalid JSON Output", state="error")
                except ValueError as ve: # Catch our validation error
                     status.write(f"Error: Agent returned JSON with incorrect structure: {ve}. See raw output below:")
                     st.code(final_output_text, language="json") # Assume it might be JSON, just wrong structure
                     status.update(label="Agent Error: Incorrect JSON Structure", state="error")
                except Exception as e:
                    status.write(f"Error processing agent output: {e}")
                    st.code(traceback.format_exc())
                    status.update(label="Agent Error: Processing Failed", state="error")
            else:
                status.update(label="Agent failed to run or returned no output.", state="error")


# --- 2. Display Curated Results ---
st.header("2. Curated Results")

if st.session_state.table_desc:
    st.subheader("Generated Table Description")
    st.markdown(f"> {st.session_state.table_desc}")

    st.subheader("Generated Column Descriptions")
    # Only display dataframe if it's not empty AND there wasn't a schema error indicated in the table description
    if not st.session_state.column_desc_df.empty and not (isinstance(st.session_state.table_desc, str) and "schema" in st.session_state.table_desc.lower() and "error" in st.session_state.table_desc.lower()):
        st.dataframe(st.session_state.column_desc_df, use_container_width=True,
                     column_config={
                         "column_name": st.column_config.TextColumn("Column Name", width="medium"),
                         "data_type": st.column_config.TextColumn("Data Type", width="small"),
                         "description": st.column_config.TextColumn("Generated Description", width="large")
                     })
    elif not (isinstance(st.session_state.table_desc, str) and "error" in st.session_state.table_desc.lower()):
         st.warning("Column descriptions could not be generated or schema was unavailable/empty.")
    # If table desc showed error, column display is implicitly skipped or warned about already
else:
    st.info("Profile a table above to see the results here.")