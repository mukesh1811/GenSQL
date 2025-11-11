from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .context import ContextPayload


class ConversationMessage(BaseModel):
    role: str
    content: Optional[str] = None
    plan_text: Optional[str] = None
    display_content: Optional[str] = None


class PlanRequest(BaseModel):
    context: ContextPayload
    conversation: List[ConversationMessage]


class PlanResponse(BaseModel):
    plan: str


class SQLRequest(BaseModel):
    context: ContextPayload
    conversation: List[ConversationMessage]
    plan: str


class SQLResponse(BaseModel):
    sql: str


class QueryRequest(BaseModel):
    sql: str


class QueryResult(BaseModel):
    columns: List[str]
    rows: List[dict]
    total_rows: int


class QueryResponse(QueryResult):
    preview: List[dict]


class ChartSuggestionRequest(BaseModel):
    user_prompt: str
    sql: str
    rows: List[dict]


class ChartSuggestion(BaseModel):
    chart_type: str
    x_axis: Optional[str]
    y_axis: Optional[str]
    title: Optional[str]


class SummaryRequest(BaseModel):
    rows: List[dict]
    user_question: Optional[str] = None


class SummaryResponse(BaseModel):
    summary: str


class TableDescriptionRequest(BaseModel):
    project: str
    dataset: str
    table: str
    description: Optional[str] = None
    columns: List[dict]


class ColumnDescriptionsRequest(BaseModel):
    project: str
    dataset: str
    table: str
    columns: List[dict]


class ColumnDescriptionsResponse(BaseModel):
    descriptions: dict[str, str]


class ExecuteWorkflowRequest(BaseModel):
    context: ContextPayload
    conversation: List[ConversationMessage]
    user_prompt: str


class WorkflowStep(BaseModel):
    kind: str = Field(description="plan|sql|query|summary|chart")
    payload: dict


class ExecuteWorkflowResponse(BaseModel):
    steps: List[WorkflowStep]
