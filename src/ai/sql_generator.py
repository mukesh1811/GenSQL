"""
SQL generation and chart suggestion using LLM.

Provides functions to generate query plans, SQL queries, and chart suggestions
based on user questions and table context.
"""

import json
import streamlit as st
import pandas as pd

from .vertex_client import get_model


def _get_llm_context() -> str | None:
    """
    Builds the context string for LLM prompts from session state.
    
    Combines information about all selected tables including their schemas
    and descriptions.
    
    Returns:
        str | None: Formatted context string, or None if context not set
    """
    if not st.session_state.selected_tables:
        st.error("Context is not set. Cannot generate SQL.")
        return None

    context_parts = []
    for idx, table_info in enumerate(st.session_state.selected_tables):
        project = table_info["project"]
        dataset = table_info["dataset"]
        table = table_info["table"]
        table_fqn = f"{project}.{dataset}.{table}"
        
        table_description = table_info.get("description") or "No description provided."
        
        # Use schema from dict if available, else look in legacy/fallback
        schema_df = st.session_state.schemas.get(table_fqn, pd.DataFrame())
        
        if schema_df.empty and idx == 0 and not st.session_state.schema_df.empty:
            # Fallback: if it's the first table and it's in the legacy var
            schema_df = st.session_state.schema_df

        if not schema_df.empty:
            schema_str = schema_df[["column_name", "data_type", "column_description"]].to_string(index=False)
        else:
            schema_str = "(Schema not available)"

        context_parts.append(f"""
**Table {idx + 1} Context:**
- Fully Qualified Table Name: `{table_fqn}`
- Table Description: {table_description}
- Table Schema:
{schema_str}
""")

    return "\n---\n".join(context_parts)


def generate_plan(conversation_text: str) -> str | None:
    """
    Generates a query plan based on conversation history and table context.
    
    Args:
        conversation_text: Formatted conversation history
        
    Returns:
        str | None: Generated plan text, or None if error occurred
    """
    model = get_model()
    context_str = _get_llm_context()
    if not context_str:
        return "Error: Context not set."

    prompt = f"""
You are an expert Google BigQuery SQL analyst. Your task is to create a step-by-step plan to answer the user's question based on the provided table context and conversation history.

{context_str}

**Conversation History:**
{conversation_text}

**Instructions:**
1. Analyze the latest user request in the context of the full conversation.
2. Clearly state the goal of the analysis in one concise sentence before listing the steps.
3. Generate a new, concise, step-by-step plan. If a previous plan exists, refine it based on the user's latest feedback.
4. The plan should be clear, unambiguous and easy to understand.
5. If the question cannot be answered using the schema, the plan must explicitly state why.
6. Output only the goal statement and the plan, with the goal as a sentence followed by a numbered or bulleted list. Do not include any preamble, titles, or markdown formatting.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"An error occurred while communicating with the AI model: {e}")
        return None


def generate_sql_from_plan(conversation_text: str, approved_plan: str) -> str | None:
    """
    Generates SQL query from an approved plan and conversation history.
    
    Args:
        conversation_text: Formatted conversation history
        approved_plan: The user-approved plan text
        
    Returns:
        str | None: Generated SQL query, or None if error occurred
    """
    model = get_model()
    context_str = _get_llm_context()
    if not context_str:
        return "Error: Context not set."
        
    prompt = f"""
You are an expert Google BigQuery SQL writer. Your task is to write a single, syntactically correct Google BigQuery SQL query based on the approved plan and table context.

{context_str}

**Full Conversation History (for context):**
{conversation_text}

**Approved Plan:**
---
{approved_plan}
---

**Instructions:**
1. Adhere strictly to the approved plan to write the query.
2. Write a single Google BigQuery SQL query.
3. If the plan states that the question is unanswerable, output the text "Unable to generate".
4. Output **only the SQL code**, without any preamble, explanation, or markdown formatting such as ```sql.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"An error occurred while communicating with the AI model: {e}")
        return None


def suggest_chart(user_prompt: str, sql_query: str, result_df: pd.DataFrame) -> dict:
    """
    Analyzes query results and suggests a suitable chart type.
    
    Args:
        user_prompt: Original user question
        sql_query: The SQL query that was executed
        result_df: Query results as DataFrame
        
    Returns:
        dict: Chart suggestion with keys: chart_type, x_axis, y_axis, title
    """
    if result_df.empty or len(result_df.columns) < 2:
        return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}

    model = get_model()
    data_preview = result_df.head().to_string()
    
    prompt = f"""
You are a data visualization expert. Based on the user's question, the SQL query, and a preview of the result data, determine the most suitable type of chart to visualize the answer.

**User's Question:** "{user_prompt}"

**SQL Query:**
```sql
{sql_query}
```

**Data Preview (first 5 rows):**
```
{data_preview}
```

**Instructions:**
1.  Analyze the data's structure (column names, number of columns, data types). A numeric column is generally required for the y-axis. A categorical or temporal column is best for the x-axis.
2.  Consider the user's intent (e.g., comparison, trend over time, distribution).
3.  Choose one of the following chart types: 'bar', 'line', 'area', 'scatter', or 'none'.
4.  If you choose a chart type other than 'none', you MUST identify the best columns for the 'x_axis' and 'y_axis' from the data preview.
5.  Provide a descriptive title for the chart.
6.  Return a single JSON object with the following keys: "chart_type", "x_axis", "y_axis", "title". If no chart is suitable, `chart_type` must be "none".

**JSON Output:**
"""
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        
        try:
            chart_info = json.loads(json_str)
            if "chart_type" in chart_info:
                return chart_info
            return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}
        except json.JSONDecodeError:
            return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}
            
    except Exception:
        return {"chart_type": "none", "x_axis": None, "y_axis": None, "title": None}


def summarize_query_result(result_df: pd.DataFrame, user_question: str | None = None) -> str:
    """
    Summarize query results using LLM to provide key insights.
    
    Args:
        result_df: Query results as DataFrame
        user_question: Original user question (optional)
        
    Returns:
        str: One key insight in plain English
    """
    if result_df is None or result_df.empty:
        return "No data available to summarize."

    # Prepare a compact preview to send to the model (limit rows & columns)
    try:
        preview = result_df.head(20).copy()
        # If too many columns, summarize the shape
        if len(preview.columns) > 10:
            col_list = ", ".join(preview.columns[:10].tolist()) + f", ... (total {len(preview.columns)} columns)"
        else:
            col_list = ", ".join(preview.columns.tolist())
        
        preview_str = preview.to_string(max_rows=20, max_cols=10)
        
        question_context = f"\n\n**User's Original Question:** {user_question}" if user_question else ""
        
        prompt = f"""
You are a data analyst. Provide ONE key insight from the following query result in plain English. 
Keep it concise (1-2 sentences). If a user question is provided, make sure the insight is relevant to their question.{question_context}

**Data Columns:** {col_list}
**Data Preview (up to 20 rows):**
{preview_str}

**Key Insight:**
"""
        
        model = get_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        # Fallback to heuristic summary
        row_count = len(result_df)
        col_count = len(result_df.columns)
        
        if user_question:
            return f"Query returned {row_count} row(s) with {col_count} column(s) related to: '{user_question}'."
        return f"Query returned {row_count} row(s) with {col_count} column(s)."
