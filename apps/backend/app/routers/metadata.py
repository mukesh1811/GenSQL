from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.metadata import DatasetListResponse, ProjectListResponse, SchemaResponse, TableListResponse
from app.services import gcp, schema

router = APIRouter(prefix="/metadata", tags=["metadata"])


@router.get("/projects", response_model=ProjectListResponse)
def list_projects():
    try:
        projects = gcp.list_projects()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ProjectListResponse(projects=projects)


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(project: str = Query(..., description="GCP project id")):
    try:
        datasets = gcp.list_datasets(project)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return DatasetListResponse(datasets=datasets)


@router.get("/tables", response_model=TableListResponse)
def list_tables(project: str = Query(...), dataset: str = Query(...)):
    try:
        tables = gcp.list_tables(project, dataset)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return TableListResponse(tables=tables)


@router.get("/schema", response_model=SchemaResponse)
def get_schema(project: str = Query(...), dataset: str = Query(...), table: str = Query(...)):
    try:
        schema_df = schema.get_table_schema(project, dataset, table)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return SchemaResponse(columns=schema.dataframe_to_records(schema_df))
