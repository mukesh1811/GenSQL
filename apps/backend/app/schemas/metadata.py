from __future__ import annotations

from typing import List

from pydantic import BaseModel


class ProjectListResponse(BaseModel):
    projects: List[str]


class DatasetListResponse(BaseModel):
    datasets: List[str]


class TableListResponse(BaseModel):
    tables: List[str]


class SchemaColumn(BaseModel):
    table_catalog: str | None = None
    table_schema: str | None = None
    table_name: str | None = None
    column_name: str
    data_type: str | None = None
    column_description: str | None = None


class SchemaResponse(BaseModel):
    columns: List[SchemaColumn]
