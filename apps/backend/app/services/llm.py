"""High level LLM orchestration utilities."""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd

from app.services import context_store, schema, vertex


def _format_context(project: str, dataset: str, table: str, description: str | None, columns: Iterable[dict]) -> str:
    columns_str = "\n".join(
        [f"- {col['column_name']} ({col['data_type']}): {col.get('column_description', '')}" for col in columns]
    )
    description = description or ""
    return (
        f"Table: {project}.{dataset}.{table}\n"
        f"Description: {description}\n"
        f"Schema:\n{columns_str}"
    )


def build_conversation_text(messages: Iterable[dict]) -> str:
    history = []
    for message in messages:
        role = message.get("role", "user").capitalize()
        content = message.get("content") or message.get("plan_text") or message.get("display_content") or ""
        history.append(f"{role}: {content}")
    return "\n".join(history)


def generate_plan(context: dict, messages: list[dict]) -> str:
    context_blob = _format_context(
        context["project"], context["dataset"], context["table"], context.get("description"), context.get("columns", [])
    )
    conversation_text = build_conversation_text(messages)
    prompt = f"""
You are an expert Google BigQuery analyst. Your task is to create a step-by-step plan to answer the user's question based on the provided table context and conversation history.

{context_blob}

Conversation History:
{conversation_text}

Instructions:
1. Analyze the latest user request in the context of the full conversation.
2. Clearly state the goal of the analysis in one concise sentence before listing the steps.
3. Generate a new, concise, step-by-step plan. If a previous plan exists, refine it based on the user's latest feedback.
4. The plan should be clear and unambiguous.
5. If the question cannot be answered using the schema, the plan must explicitly state why.
6. Output only the goal statement and the plan as a numbered list. Do not include any preamble.
"""
    return vertex.generate_content(prompt)


def generate_sql(context: dict, conversation: list[dict], plan_text: str) -> str:
    context_blob = _format_context(
        context["project"], context["dataset"], context["table"], context.get("description"), context.get("columns", [])
    )
    conversation_text = build_conversation_text(conversation)
    prompt = f"""
You are an expert Google BigQuery SQL writer. Write a single, syntactically correct query based on the approved plan and context.

{context_blob}

Conversation History:
{conversation_text}

Approved Plan:
{plan_text}

Instructions:
1. Follow the approved plan strictly.
2. Output only the SQL code without markdown fences.
3. If the plan indicates the task is impossible, respond with "Unable to generate".
"""
    return vertex.generate_content(prompt)


def enhance_table_description(project: str, dataset: str, table: str, schema_df: pd.DataFrame, current_description: str | None = None) -> str:
    schema_lines = "\n".join([f"- {row.column_name} ({row.data_type})" for _, row in schema_df.iterrows()])
    prompt = f"""
Based on the fully qualified table name `{project}.{dataset}.{table}` and its schema, provide a concise one sentence description.

Existing description: {current_description}
Schema:
{schema_lines}

Description:
"""
    return vertex.generate_content(prompt)


def enhance_column_descriptions(project: str, dataset: str, table: str, schema_df: pd.DataFrame) -> dict[str, str]:
    summaries = schema.compute_column_summaries(project, dataset, table, schema_df)
    column_meta = schema_df[["column_name", "data_type"]].to_dict(orient="records")
    summaries_str = json.dumps(summaries, default=str)
    prompt = f"""
Given the table `{project}.{dataset}.{table}` and the following per-column metadata, provide a concise description for each column.

Columns:
{column_meta}

Summaries:
{summaries_str}

Return a JSON object mapping column names to descriptions.
"""
    response = vertex.generate_content(prompt)
    cleaned = response.removeprefix("```json").removesuffix("```").strip()
    try:
        descriptions = json.loads(cleaned)
    except json.JSONDecodeError:
        descriptions = {}
    enhanced: dict[str, str] = {}
    for column in schema_df["column_name"].tolist():
        base_desc = descriptions.get(column, "")
        summary_suffix = schema.humanize_summary(column, summaries.get(column, {}))
        enhanced[column] = f"{base_desc}{summary_suffix}".strip()
    return enhanced


def suggest_chart(user_prompt: str, sql_query: str, result_df: pd.DataFrame) -> dict:
    if result_df.empty or len(result_df.columns) < 2:
        return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}
    preview = result_df.head().to_string()
    prompt = f"""
You are a data visualization expert. Based on the user's question, the SQL query, and a preview of the result data, determine the best chart.

User question: {user_prompt}
SQL Query:
{sql_query}

Data Preview:
{preview}

Return a JSON object with keys chart_type, x_axis, y_axis, title. Chart type must be one of bar, line, area, scatter, or none.
"""
    response = vertex.generate_content(prompt)
    cleaned = response.removeprefix("```json").removesuffix("```").strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = {}
    if "chart_type" not in parsed:
        return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}
    return parsed


def summarize_dataframe(result_df: pd.DataFrame, user_question: str | None = None) -> str:
    if result_df.empty:
        return "No data available to summarize."
    preview = result_df.head(20)
    schema_lines = [f"- {col}: {preview[col].dtype}" for col in preview.columns]
    preview_csv = preview.to_csv(index=False)
    question = f"User question: {user_question}\n" if user_question else ""
    prompt = f"""
You are a data analyst. Given the result preview, column types, and the user's question, provide one key insight.

{question}Column types:
{chr(10).join(schema_lines)}

Data preview (CSV):
{preview_csv}

Insight:
"""
    try:
        return vertex.generate_content(prompt)
    except Exception:
        numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            series = result_df[numeric_cols[0]]
            return (
                f"Key insight: Column '{numeric_cols[0]}' ranges from {series.min()} to {series.max()} with an average of {series.mean():.2f}."
            )
        column = result_df.columns[0]
        mode = result_df[column].mode()
        if not mode.empty:
            value = mode.iloc[0]
            count = (result_df[column] == value).sum()
            return f"Key insight: '{column}' most commonly has the value '{value}' ({count} occurrences)."
        return "Query executed. No summary available."


def schema_context_blob(context: dict) -> str:
    return _format_context(
        context["project"], context["dataset"], context["table"], context.get("description"), context.get("columns", [])
    )
