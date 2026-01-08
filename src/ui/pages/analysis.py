"""Main analysis page (Chat interface)"""

import streamlit as st
import pandas as pd

from src.utils.chat import build_llm_conversation_text
from src.database import run_query
from src.ai.sql_generator import (
    generate_plan,
    generate_sql_from_plan,
    suggest_chart,
    summarize_query_result
)

def render_analysis_page():
    if not st.session_state.ctx_set:
        st.markdown(
            "<div style='background-color:#fff3cd;border:1px solid #ffeeba;color:#856404;padding:12px;border-radius:4px;text-align:center;'>"
            "<span style='font-size:1.05em;'>⚠️ Context not set. Please set context in the left sidebar to proceed.</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    
    st.markdown("<h2 style='text-align:center;margin-top:0;'>🧐 What are you analyzing today?</h2>", unsafe_allow_html=True)
    
    if not st.session_state.ctx_set:
        st.chat_input("Set context to start analyzing your data.", disabled=True)
        return
    
    st.write("")

    # Find the index of the last message that is an unapproved plan
    last_plan_idx = -1
    for i in range(len(st.session_state.messages) - 1, -1, -1):
        msg = st.session_state.messages[i]
        if msg.get("type") == "plan" and not msg.get("approved", False):
            last_plan_idx = i
            break
            
    # Display all messages
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["display_content"])

            if msg.get("type") == "sql":
                if "query_result" in msg:
                    st.caption("Query Result:")
                    st.dataframe(msg["query_result"])

                    # If a summary was previously generated for this result, display it
                    if "summary" in msg:
                        st.info(msg["summary"])

                    # Summarize button
                    if st.button("📝 Summarize", key=f"summarize_{i}"):
                        full_df = msg.get("full_query_result", msg.get("query_result"))
                        # Attempt to find the original user question
                        original_prompt = ""
                        for j in range(i, -1, -1):
                            if st.session_state.messages[j]["role"] == "user":
                                original_prompt = st.session_state.messages[j]["content"]
                                break
                        summary_text = summarize_query_result(full_df, user_question=original_prompt)
                        st.session_state.messages[i]["summary"] = summary_text
                        st.rerun()
                    
                    if msg.get("chart_created"):
                        st.caption("Chart:")
                        chart_info = msg["chart_suggestion"]
                        chart_df = msg["full_query_result"] # Use full result for charting
                        
                        # Verify columns exist before trying to chart
                        if chart_info["x_axis"] in chart_df.columns and chart_info["y_axis"] in chart_df.columns:
                            try:
                                chart_df_indexed = chart_df.set_index(chart_info["x_axis"])
                            except Exception:
                                chart_df_indexed = chart_df
                                
                            chart_type = chart_info["chart_type"]
                            st.write(f"**{chart_info.get('title', 'Generated Chart')}**")

                            if chart_type == "bar": st.bar_chart(chart_df_indexed[[chart_info["y_axis"]]])
                            elif chart_type == "line": st.line_chart(chart_df_indexed[[chart_info["y_axis"]]])
                            elif chart_type == "area": st.area_chart(chart_df_indexed[[chart_info["y_axis"]]])
                            elif chart_type == "scatter": st.scatter_chart(chart_df, x=chart_info["x_axis"], y=chart_info["y_axis"])
                        else:
                            st.error("Chart generation failed: columns suggested by AI were not found in the result.")

                    elif "chart_suggestion" in msg and msg["chart_suggestion"]["chart_type"] != "none":
                        chart_info = msg["chart_suggestion"]
                        chart_type_str = chart_info['chart_type'].capitalize()
                        if st.button(f"📊 Create {chart_type_str} Chart", key=f"create_chart_{i}"):
                            st.session_state.messages[i]["chart_created"] = True
                            st.rerun()

                elif msg.get("sql_text") and "Unable to generate" not in msg["sql_text"]:
                    # SQL Manual Editing Logic
                    if msg.get("editable"):
                        current_sql = msg.get("edited_sql", msg.get("sql_text", ""))
                        edited = st.text_area("Edit SQL", value=current_sql, key=f"manual_sql_editor_{i}", height=220)
                        c_upd, c_can = st.columns([1, 1])
                        if c_upd.button("Update", key=f"update_sql_{i}"):
                            st.session_state.messages[i]["sql_text"] = edited
                            st.session_state.messages[i]["display_content"] = f"**Generated SQL:**\n```sql\n{edited}\n```"
                            st.session_state.messages[i].pop("editable", None)
                            st.session_state.messages[i].pop("edited_sql", None)
                            st.rerun()
                        if c_can.button("Cancel", key=f"cancel_edit_{i}"):
                            st.session_state.messages[i].pop("editable", None)
                            st.session_state.messages[i].pop("edited_sql", None)
                            st.rerun()
                        st.session_state.messages[i]["edited_sql"] = edited
                    else:
                        c_run, c_edit = st.columns([1, 1])
                        if c_edit.button("✏️ Manual Edit", key=f"manual_edit_{i}"):
                            st.session_state.messages[i]["editable"] = True
                            st.session_state.messages[i]["edited_sql"] = msg.get("sql_text", "")
                            st.rerun()

                        if c_run.button("🚀 Run SQL", key=f"run_sql_{i}"):
                            result_df = run_query(msg["sql_text"])
                            if result_df is not None:
                                st.session_state.messages[i]["full_query_result"] = result_df
                                
                                if len(result_df) > 10:
                                    st.warning(f"Displaying the first 10 of {len(result_df)} rows.")
                                    st.session_state.messages[i]["query_result"] = result_df.head(10)
                                else:
                                    st.session_state.messages[i]["query_result"] = result_df
                                
                                # Suggest chart
                                with st.spinner("Analyzing result for chart suggestion..."):
                                    original_prompt = ""
                                    for j in range(i, -1, -1):
                                        if st.session_state.messages[j]["role"] == "user":
                                            original_prompt = st.session_state.messages[j]["content"]
                                            break
                                    chart_suggestion = suggest_chart(original_prompt, msg["sql_text"], result_df)
                                    if chart_suggestion:
                                        st.session_state.messages[i]["chart_suggestion"] = chart_suggestion
                                st.rerun()

    # Approve Plan Logic
    if last_plan_idx != -1 and last_plan_idx == len(st.session_state.messages) - 1:
        st.session_state.current_plan_approved = False
        if st.button("✅ Approve Plan"):
            st.session_state.messages[last_plan_idx]["approved"] = True
            
            with st.chat_message("assistant"):
                with st.spinner("Generating SQL from approved plan..."):
                    approved_plan = st.session_state.messages[last_plan_idx]["plan_text"]
                    history_for_sql = st.session_state.messages[:last_plan_idx + 1]
                    conversation_text = build_llm_conversation_text(history_for_sql)
                    
                    sql_code = generate_sql_from_plan(conversation_text, approved_plan)

                    if sql_code:
                        if "Unable to generate" in sql_code:
                            display_content = "⚠️ As per the plan, I am unable to generate the SQL for this request."
                        else:
                            display_content = f"**Generated SQL:**\n```sql\n{sql_code}\n```"

                        st.session_state.messages.append({
                            "role": "assistant", "type": "sql",
                            "sql_text": sql_code, "display_content": display_content
                        })
            st.rerun()

    # Chat Input Logic
    is_mid_conversation = last_plan_idx != -1 and last_plan_idx == len(st.session_state.messages) - 1
    placeholder = "Suggest an edit to the plan, or approve it." if is_mid_conversation else "What are you analyzing today?"
    
    if prompt := st.chat_input(placeholder, disabled=not st.session_state.ctx_set):
        st.session_state.messages.append({"role": "user", "display_content": prompt, "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                conversation_text = build_llm_conversation_text(st.session_state.messages)
                new_plan_text = generate_plan(conversation_text)

                if new_plan_text:
                    display_content = f"**Here is the goal and proposed plan:**\n\n{new_plan_text}"
                    st.session_state.messages.append({
                        "role": "assistant", "type": "plan",
                        "plan_text": new_plan_text, "display_content": display_content,
                        "approved": False
                    })
        st.rerun()
