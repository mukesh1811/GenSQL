from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.schemas.analytics import (
    ChartSuggestion,
    ChartSuggestionRequest,
    ColumnDescriptionsRequest,
    ColumnDescriptionsResponse,
    ExecuteWorkflowRequest,
    ExecuteWorkflowResponse,
    PlanRequest,
    PlanResponse,
    QueryRequest,
    QueryResponse,
    SQLRequest,
    SQLResponse,
    SummaryRequest,
    SummaryResponse,
    TableDescriptionRequest,
)
from app.services import llm, schema
from app.services.gcp import query_to_dataframe

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/plan", response_model=PlanResponse)
def create_plan(payload: PlanRequest):
    try:
        plan = llm.generate_plan(payload.context.dict(), [msg.dict() for msg in payload.conversation])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return PlanResponse(plan=plan)


@router.post("/sql", response_model=SQLResponse)
def generate_sql(payload: SQLRequest):
    try:
        sql = llm.generate_sql(payload.context.dict(), [msg.dict() for msg in payload.conversation], payload.plan)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return SQLResponse(sql=sql)


@router.post("/query", response_model=QueryResponse)
def execute_query(payload: QueryRequest):
    try:
        df = query_to_dataframe(payload.sql)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    preview = df.head(10).to_dict(orient="records")
    rows = df.to_dict(orient="records")
    return QueryResponse(columns=df.columns.tolist(), rows=rows, total_rows=len(df), preview=preview)


@router.post("/chart", response_model=ChartSuggestion)
def suggest_chart(payload: ChartSuggestionRequest):
    df = pd.DataFrame(payload.rows)
    try:
        suggestion = llm.suggest_chart(payload.user_prompt, payload.sql, df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ChartSuggestion(**suggestion)


@router.post("/summary", response_model=SummaryResponse)
def summarize(payload: SummaryRequest):
    df = pd.DataFrame(payload.rows)
    try:
        summary = llm.summarize_dataframe(df, payload.user_question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return SummaryResponse(summary=summary)


@router.post("/describe/table", response_model=SummaryResponse)
def enhance_table(payload: TableDescriptionRequest):
    df = pd.DataFrame(payload.columns)
    try:
        text = llm.enhance_table_description(payload.project, payload.dataset, payload.table, df, payload.description)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return SummaryResponse(summary=text)


@router.post("/describe/columns", response_model=ColumnDescriptionsResponse)
def enhance_columns(payload: ColumnDescriptionsRequest):
    df = pd.DataFrame(payload.columns)
    try:
        descriptions = llm.enhance_column_descriptions(payload.project, payload.dataset, payload.table, df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ColumnDescriptionsResponse(descriptions=descriptions)


@router.post("/workflow", response_model=ExecuteWorkflowResponse)
def run_workflow(payload: ExecuteWorkflowRequest):
    steps = []
    try:
        plan = llm.generate_plan(payload.context.dict(), [msg.dict() for msg in payload.conversation])
        steps.append({"kind": "plan", "payload": {"plan": plan}})
        sql = llm.generate_sql(payload.context.dict(), [msg.dict() for msg in payload.conversation], plan)
        steps.append({"kind": "sql", "payload": {"sql": sql}})
        df = query_to_dataframe(sql)
        steps.append({"kind": "query", "payload": {"rows": df.to_dict(orient="records"), "columns": df.columns.tolist()}})
        summary = llm.summarize_dataframe(df, payload.user_prompt)
        steps.append({"kind": "summary", "payload": {"summary": summary}})
        chart = llm.suggest_chart(payload.user_prompt, sql, df)
        steps.append({"kind": "chart", "payload": chart})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return ExecuteWorkflowResponse(steps=steps)
