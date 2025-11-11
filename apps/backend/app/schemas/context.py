from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ColumnSchema(BaseModel):
    table_catalog: Optional[str] = None
    table_schema: Optional[str] = None
    table_name: Optional[str] = None
    column_name: str
    data_type: Optional[str] = None
    column_description: Optional[str] = Field(default=None, description="Human friendly column description")


class ContextPayload(BaseModel):
    project: str
    dataset: str
    table: str
    description: Optional[str] = None
    columns: List[ColumnSchema]


class ContextResponse(ContextPayload):
    source: Optional[str] = Field(default=None, description="Context origin e.g. upload or bigquery")


class ContextSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)


class ContextSearchResult(BaseModel):
    metadata: dict
    distance: float
    document: str


class PersistContextRequest(ContextPayload):
    source: str = Field(default="upload")
