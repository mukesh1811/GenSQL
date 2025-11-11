"""Helpers for interacting with Google Cloud resources."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import google.auth
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery
from google.cloud.bigquery import DatasetListItem, Project


class GCPError(RuntimeError):
    """Raised when a Google Cloud operation fails."""


@lru_cache(maxsize=1)
def get_bigquery_client() -> bigquery.Client:
    """Create a cached BigQuery client using Application Default Credentials."""

    return bigquery.Client()


def get_active_project() -> str:
    """Return the active project id from BigQuery client or ADC."""

    try:
        return get_bigquery_client().project
    except Exception:  # pragma: no cover - fallback path
        _, project_id = google.auth.default()
        if not project_id:
            raise GCPError("Could not determine the active GCP project. Configure ADC credentials.")
        return project_id


def list_projects() -> list[str]:
    """List accessible project IDs."""

    client = get_bigquery_client()
    projects: Iterable[Project] = client.list_projects()
    return [project.project_id for project in projects]


def list_datasets(project_id: str) -> list[str]:
    """List dataset IDs for a project."""

    client = get_bigquery_client()
    datasets: Iterable[DatasetListItem] = client.list_datasets(project=project_id)
    return [dataset.dataset_id for dataset in datasets]


def list_tables(project_id: str, dataset_id: str) -> list[str]:
    """List table IDs for a dataset."""

    client = get_bigquery_client()
    tables = client.list_tables(f"{project_id}.{dataset_id}")
    return [table.table_id for table in tables]


def fetch_table_schema(project_id: str, dataset_id: str, table_id: str) -> bigquery.table.Table:
    """Return the BigQuery table object for schema inspection."""

    client = get_bigquery_client()
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    try:
        return client.get_table(table_ref)
    except GoogleAPIError as exc:  # pragma: no cover - pass through API errors
        raise GCPError(str(exc)) from exc


def query_to_dataframe(sql: str):
    """Execute SQL against BigQuery and return the result DataFrame."""

    client = get_bigquery_client()
    try:
        return client.query(sql).to_dataframe()
    except GoogleAPIError as exc:
        raise GCPError(str(exc)) from exc
