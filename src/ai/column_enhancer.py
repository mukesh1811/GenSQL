"""
Column and table description enhancement using LLM.

Provides functions to auto-generate descriptions for tables and columns
based on their names, schemas, and actual data statistics from BigQuery.
"""

import json
import streamlit as st
import pandas as pd

from .vertex_client import get_model
from src.database import get_bq_client


def enhance_table_description() -> None:
    """
    Uses the LLM to generate a table description based on its name and schema.
    
    Updates st.session_state.bq_table_desc with the generated description.
    Requires: pick_project, pick_dataset, pick_table, bq_schema_df in session state.
    """
    model = get_model()
    project = st.session_state.pick_project
    dataset = st.session_state.pick_dataset
    table = st.session_state.pick_table
    schema_df = st.session_state.bq_schema_df

    if not all([project, dataset, table]) or schema_df.empty:
        st.warning("Please select a valid table first.")
        return

    schema_str = "\n".join(
        [f"- {row.column_name} ({row.data_type})" for _, row in schema_df.iterrows()]
    )
    
    current_desc = st.session_state.get("bq_table_desc", "")
    
    prompt = f"""
    Based on the fully qualified table name `{project}.{dataset}.{table}` and its schema, please provide a concise, one-sentence description of what this table likely contains.
    Consider this table description if given: {current_desc}
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


def enhance_column_descriptions() -> None:
    """
    Uses the LLM to generate descriptions for all columns in a table.
    
    Analyzes actual data from BigQuery to get statistics (min/max, distinct values, etc.)
    and uses those to generate meaningful column descriptions.
    
    Updates st.session_state.bq_schema_df with enhanced descriptions.
    Requires:  pick_project, pick_dataset, pick_table, bq_schema_df, bq_table_desc in session state.
    """
    model = get_model()
    project = st.session_state.pick_project
    dataset = st.session_state.pick_dataset
    table = st.session_state.pick_table
    schema_df = st.session_state.bq_schema_df.copy()

    if not all([project, dataset, table]) or schema_df.empty:
        st.warning("Please select a valid table first.")
        return

    client = get_bq_client()

    # Collect lightweight summaries for each column
    summaries = {}
    with st.spinner("Analyzing column statistics in BigQuery..."):
        for _, row in schema_df.iterrows():
            col = row['column_name']
            dtype = (row.get('data_type') or '').upper()
            col_back = f"`{col}`"
            table_fq = f"`{project}.{dataset}.{table}`"

            # Build targeted query based on data type
            if any(t in dtype for t in ['STRING', 'BYTES', 'CHAR']):
                qry = f"SELECT ARRAY_AGG(DISTINCT {col_back} ORDER BY {col_back} LIMIT 10) AS distinct_vals, COUNT(DISTINCT {col_back}) AS distinct_count FROM {table_fq} WHERE {col_back} IS NOT NULL"
            elif any(t in dtype for t in ['DATE', 'TIMESTAMP', 'DATETIME']):
                qry = f"SELECT MIN({col_back}) AS min_val, MAX({col_back}) AS max_val FROM {table_fq} WHERE {col_back} IS NOT NULL"
            elif any(t in dtype for t in ['INT', 'INTEGER', 'NUMERIC', 'FLOAT', 'DOUBLE', 'DECIMAL']):
                qry = f"SELECT MIN({col_back}) AS min_val, MAX({col_back}) AS max_val, AVG({col_back}) AS avg_val FROM {table_fq} WHERE {col_back} IS NOT NULL"
            elif any(t in dtype for t in ['BOOL', 'BOOLEAN']):
                qry = f"SELECT COUNTIF({col_back}) AS true_count, COUNT(*) - COUNTIF({col_back}) AS false_count, COUNT(*) AS total_count FROM {table_fq}"
            else:
                # Fallback: sample distinct values
                qry = f"SELECT ARRAY_AGG(DISTINCT {col_back} ORDER BY {col_back} LIMIT 10) AS distinct_vals, COUNT(DISTINCT {col_back}) AS distinct_count FROM {table_fq} WHERE {col_back} IS NOT NULL"

            try:
                df_sum = client.query(qry).to_dataframe()
                if not df_sum.empty:
                    summaries[col] = {
                        k: (v.tolist() if hasattr(v, 'tolist') else v)
                        for k, v in df_sum.iloc[0].to_dict().items()
                    }
                else:
                    summaries[col] = {}
            except Exception as e:
                summaries[col] = {"error": str(e)}

    # Build prompt with summaries
    cols_to_describe = schema_df[['column_name', 'data_type']].to_dict('records')
    try:
        summaries_str = json.dumps(summaries, default=str)
    except Exception:
        summaries_str = str(summaries)

    table_desc = st.session_state.get("bq_table_desc", "")
    
    prompt = f"""
Given the table name `{project}.{dataset}.{table}`, and the following per-column lightweight data summaries, provide a concise, one-line description for each of the columns.
Consider this table description if given: {table_desc}
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
                st.warning("AI response was not valid JSON, falling back to heuristic summary.")
                desc_dict = {}

            # Build human-readable appendices from summaries
            def humanize_summary(col_name: str, summ: dict) -> str:
                if not summ:
                    return ""
                if 'error' in summ:
                    return " (Could not compute data summary)"
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

            # Generate final descriptions
            final_descriptions = {}
            for _, row in schema_df.iterrows():
                col = row['column_name']
                base = ''
                if isinstance(desc_dict, dict) and col in desc_dict:
                    base = str(desc_dict[col]).strip()
                else:
                    # Fallback heuristics
                    dt = (row.get('data_type') or '').upper()
                    if any(t in dt for t in ['INT', 'NUMERIC', 'FLOAT', 'DOUBLE', 'DECIMAL']):
                        base = "Numeric field."
                    elif any(t in dt for t in ['DATE', 'TIMESTAMP', 'DATETIME']):
                        base = "Temporal field."
                    elif any(t in dt for t in ['BOOL', 'BOOLEAN']):
                        base = "Boolean flag."
                    else:
                        base = "Categorical/text field."

                appendix = humanize_summary(col, summaries.get(col, {}))
                final_descriptions[col] = (base + appendix).strip()

            # Apply to schema and update session
            descriptions = pd.Series(final_descriptions)
            schema_df['column_description'] = schema_df['column_name'].map(descriptions).fillna(
                schema_df['column_description']
            )
            st.session_state.bq_schema_df = schema_df
            st.rerun()
        except Exception as e:
            st.error(f"AI enhancement failed: {e}")
