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
        # Convert date/timestamp columns for JSON serialization if needed
        for col in df.select_dtypes(include=['datetime64[ns]', 'dbdate', 'timestamp']).columns:
            df[col] = df[col].astype(str)
        return df.to_json(orient='records')
    except Exception as e:
        return json.dumps({"error": f"Query failed for '{sql_query_limited}': {e}"})

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
        model_name="gemini-2.5-flash",
        generation_config={
            "temperature": 0.1,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 2048,
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
if "column_desc_df" not in st.session_state: st.session_state.column_desc_df = pd.DataFrame(columns=["column_name", "data_type", "description"])

st.header("1. Select a Table to Profile")
st.info("Ensure your Application Default Credentials (`gcloud auth application-default login`) have access to the selected GCP Project.", icon="🔑")

# Use public data for default example
# DEFAULT_PROJECT = "aeo-datasci-common-prod"
# DEFAULT_DATASET = "Lost_sales"
# DEFAULT_TABLE = "LOST_SALES_COMMON"

# DEFAULT_PROJECT = "bigquery-public-data"
# DEFAULT_DATASET = "samples"
# DEFAULT_TABLE = "wikipedia"

DEFAULT_PROJECT = "learning-prj-id"
DEFAULT_DATASET = "forecasting_sticker_sales"
DEFAULT_TABLE = "train"

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
4. Create descriptions that reflect the table's purpose and grain
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
        st.session_state.column_desc_df = pd.DataFrame(columns=["column_name", "data_type", "description"])

        user_input_text = f"Please profile this table: {project_id}.{dataset_id}.{table_id}"

        with st.status("🤖 **Auto-Curator Agent is working...**", expanded=True) as status:
            final_output_text = run_gemini(status, user_input_text)

            if final_output_text:
                try:
                    # Clean up markdown backticks and ensure it's just the JSON
                    clean_json_text = final_output_text.strip()
                    if clean_json_text.startswith("```json"):
                        clean_json_text = clean_json_text[7:-4].strip()
                    elif clean_json_text.startswith("```"):
                        clean_json_text = clean_json_text[3:-3].strip()

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
                        st.session_state.column_desc_df = pd.DataFrame() # Ensure empty dataframe

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