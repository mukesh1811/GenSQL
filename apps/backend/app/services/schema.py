"""Utilities for working with table schemas."""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from app.services import gcp


def get_table_schema(project: str, dataset: str, table: str) -> pd.DataFrame:
    """Return schema dataframe similar to the Streamlit prototype."""

    table_obj = gcp.fetch_table_schema(project, dataset, table)
    schema_rows = []
    for field in table_obj.schema:
        schema_rows.append(
            {
                "table_catalog": project,
                "table_schema": dataset,
                "table_name": table,
                "column_name": field.name,
                "data_type": field.field_type,
                "column_description": field.description or "<IMP: Add a brief description>",
            }
        )
    return pd.DataFrame(schema_rows)


def compute_column_summaries(project: str, dataset: str, table: str, schema_df: pd.DataFrame) -> dict[str, dict]:
    """Replicate lightweight column summaries from the prototype."""

    client = gcp.get_bigquery_client()
    table_fq = f"`{project}.{dataset}.{table}`"
    summaries: dict[str, dict] = {}
    for _, row in schema_df.iterrows():
        column = row.get("column_name")
        data_type = (row.get("data_type") or "").upper()
        if not column:
            continue
        column_backtick = f"`{column}`"
        if any(t in data_type for t in ["STRING", "BYTES", "CHAR"]):
            query = f"SELECT ARRAY_AGG(DISTINCT {column_backtick} ORDER BY {column_backtick} LIMIT 10) AS distinct_vals, COUNT(DISTINCT {column_backtick}) AS distinct_count FROM {table_fq} WHERE {column_backtick} IS NOT NULL"
        elif any(t in data_type for t in ["DATE", "TIMESTAMP", "DATETIME"]):
            query = f"SELECT MIN({column_backtick}) AS min_val, MAX({column_backtick}) AS max_val FROM {table_fq} WHERE {column_backtick} IS NOT NULL"
        elif any(t in data_type for t in ["INT", "INTEGER", "NUMERIC", "FLOAT", "DOUBLE", "DECIMAL"]):
            query = f"SELECT MIN({column_backtick}) AS min_val, MAX({column_backtick}) AS max_val, AVG({column_backtick}) AS avg_val FROM {table_fq} WHERE {column_backtick} IS NOT NULL"
        elif any(t in data_type for t in ["BOOL", "BOOLEAN"]):
            query = f"SELECT COUNTIF({column_backtick}) AS true_count, COUNT(*) - COUNTIF({column_backtick}) AS false_count, COUNT(*) AS total_count FROM {table_fq}"
        else:
            query = f"SELECT ARRAY_AGG(DISTINCT {column_backtick} ORDER BY {column_backtick} LIMIT 10) AS distinct_vals, COUNT(DISTINCT {column_backtick}) AS distinct_count FROM {table_fq} WHERE {column_backtick} IS NOT NULL"
        try:
            df = client.query(query).to_dataframe()
            summaries[column] = {
                key: (value.tolist() if hasattr(value, "tolist") else value)
                for key, value in df.iloc[0].to_dict().items()
            } if not df.empty else {}
        except Exception as exc:  # pragma: no cover - BQ failure path
            summaries[column] = {"error": str(exc)}
    return summaries


def humanize_summary(column_name: str, summary: dict) -> str:
    """Generate a concise human readable description from summary stats."""

    if not summary:
        return ""
    if "error" in summary:
        return " (Data summary unavailable)"
    if "distinct_vals" in summary or "distinct_count" in summary:
        distinct = summary.get("distinct_count")
        values = summary.get("distinct_vals")
        sample = ", ".join([str(x) for x in values[:5]]) if values else ""
        suffix = f" ({sample})" if sample else ""
        return f" (approx. {distinct} distinct values{suffix})"
    if {"min_val", "max_val", "avg_val"}.issubset(summary.keys()):
        return (
            f" (range {summary['min_val']} → {summary['max_val']}, avg {summary['avg_val']})"
        )
    if "true_count" in summary and "false_count" in summary:
        return (
            f" ({summary['true_count']} true vs {summary['false_count']} false)"
        )
    return ""


def dataframe_to_records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))
