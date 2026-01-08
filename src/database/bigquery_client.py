"""
BigQuery client and operations.

Provides cached BigQuery client and common operations like listing projects,
datasets, tables, and fetching table schemas.
"""

import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.api_core.exceptions import GoogleAPIError


@st.cache_resource(show_spinner=True)
def get_bq_client() -> bigquery.Client:
    """
    Initializes and returns a BigQuery client.
    
    Uses Application Default Credentials (ADC) from `gcloud auth application-default login`.
    The client is cached to avoid repeated initialization.
    
    Returns:
        bigquery.Client: Initialized BigQuery client
    """
    return bigquery.Client()


@st.cache_data(show_spinner=False)
def list_projects() -> list[str]:
    """
    Lists available GCP projects.
    
    Returns:
        list[str]: List of project IDs accessible to the current user
    """
    client = get_bq_client()
    return [p.project_id for p in client.list_projects()]


@st.cache_data(show_spinner=False)
def list_datasets(project_id: str) -> list[str]:
    """
    Lists datasets in a given project.
    
    Args:
        project_id: GCP project ID
        
    Returns:
        list[str]: List of dataset IDs in the project
    """
    client = get_bq_client()
    return [d.dataset_id for d in client.list_datasets(project=project_id)]


@st.cache_data(show_spinner=False)
def list_tables(project_id: str, dataset_id: str) -> list[str]:
    """
    Lists tables in a given dataset.
    
    Args:
        project_id: GCP project ID
        dataset_id: Dataset ID
        
    Returns:
        list[str]: List of table IDs in the dataset
    """
    client = get_bq_client()
    return [t.table_id for t in client.list_tables(f"{project_id}.{dataset_id}")]


@st.cache_data(show_spinner="Fetching table schema...")
def get_table_schema(project_id: str, dataset_id: str, table_id: str) -> pd.DataFrame:
    """
    Fetches the schema for a given BigQuery table.
    
    Queries INFORMATION_SCHEMA.COLUMNS to get column metadata and adds
    a column_description field (with defaults from ChromaDB if available).
    
    Args:
        project_id: GCP project ID
        dataset_id: Dataset ID
        table_id: Table ID
        
    Returns:
        pd.DataFrame: DataFrame with columns: table_catalog, table_schema,
                     table_name, column_name, data_type, column_description
    """
    from src.context.manager import get_context  # Import here to avoid circular dependency
    
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
        if stored_context["ids"]:
            for metadata in stored_context["metadatas"]:
                column_name = metadata["column_name"]
                mask = df["column_name"] == column_name
                if any(mask):
                    df.loc[mask, "column_description"] = metadata.get(
                        "description", default_desc
                    )
        
        return df
    except Exception as e:
        st.error(f"Failed to fetch schema: {e}")
        return pd.DataFrame()


def run_query(sql_query: str) -> pd.DataFrame | None:
    """
    Executes a BigQuery SQL query and returns the result as a DataFrame.
    
    Args:
        sql_query: SQL query string to execute
        
    Returns:
        pd.DataFrame | None: Query results as DataFrame, or None if error occurred
    """
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


def estimate_query_cost(sql_query: str) -> float:
    """
    Estimate the cost of running a query without actually executing it.
    
    Uses BigQuery dry_run to determine bytes that would be processed.
    
    Args:
        sql_query: SQL query string
        
    Returns:
        float: Estimated cost in USD
    """
    from src.config import settings
    
    client = get_bq_client()
    job_config = bigquery.QueryJobConfig(dry_run=True)
    
    try:
        query_job = client.query(sql_query, job_config=job_config)
        bytes_processed = query_job.total_bytes_processed
        cost_per_tb = settings.BQ_COST_PER_TB
        estimated_cost = (bytes_processed / (1024 ** 4)) * cost_per_tb
        return estimated_cost
    except Exception as e:
        st.warning(f"Could not estimate query cost: {e}")
        return 0.0
