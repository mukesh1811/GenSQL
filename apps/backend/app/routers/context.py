from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.context import (
    ContextPayload,
    ContextResponse,
    ContextSearchRequest,
    ContextSearchResult,
    PersistContextRequest,
)
from app.services import context_store

router = APIRouter(prefix="/context", tags=["context"])


@router.get("/current", response_model=ContextResponse | None)
def get_current_context():
    marker = context_store.load_current_context_marker()
    if not marker:
        return None
    table_id = marker.get("table")
    schema_data = context_store.load_context(table_id=table_id)
    records = schema_data.get("metadatas", []) if schema_data else []
    df = context_store.dataframe_from_schema_records(records)
    columns = df.to_dict(orient="records") if not df.empty else []
    return ContextResponse(
        project=marker.get("project"),
        dataset=marker.get("dataset"),
        table=table_id,
        description=marker.get("description"),
        columns=columns,
        source="persisted",
    )


@router.post("/persist", response_model=ContextResponse)
def persist_context(payload: PersistContextRequest):
    df = pd.DataFrame([column.dict() for column in payload.columns])
    try:
        context_store.persist_schema(
            project=payload.project,
            dataset=payload.dataset,
            table=payload.table,
            schema_df=df,
            table_description=payload.description,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ContextResponse(
        project=payload.project,
        dataset=payload.dataset,
        table=payload.table,
        description=payload.description,
        columns=df.to_dict(orient="records"),
        source=payload.source,
    )


@router.post("/search", response_model=list[ContextSearchResult])
def search_contexts(payload: ContextSearchRequest):
    try:
        results = context_store.search_contexts(payload.query, payload.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return [ContextSearchResult(**result) for result in results]


@router.delete("/column/{table}/{column}")
def delete_context_column(table: str, column: str):
    try:
        context_store.delete_context(table, column)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "deleted"}
