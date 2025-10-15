import streamlit as st
import pandas as pd
from datetime import date

# st.set_page_config(page_title="Streamlit Table Editor – Demo", page_icon="📝", layout="wide")
# st.title("📝 Streamlit Table Editor — Feature Showcase")

# # -----------------------------
# # Sample / Upload
# # -----------------------------
# with st.sidebar:
#     st.header("Data source")
#     uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
#     if uploaded:
#         df_src = pd.read_csv(uploaded)
#     else:
#         st.caption("No file? Using a small sample.")
#         df_src = pd.DataFrame(
#             {
#                 "id": [101, 102, 103],
#                 "name": ["Alpha", "Beta", "Gamma"],
#                 "owner": ["mukesh", "lee", "ana"],
#                 "status": ["todo", "doing", "done"],
#                 "priority": [2, 1, 3],
#                 "active": [True, True, False],
#                 "due_date": [date.today(), date.today(), date.today()],
#                 "docs": [
#                     "https://docs.streamlit.io",
#                     "https://streamlit.io",
#                     "https://discuss.streamlit.io",
#                 ],
#             }
#         )

#     st.divider()
#     st.subheader("Editor options")
#     allow_add = st.checkbox("Allow adding/removing rows", value=True)
#     use_container_width = st.checkbox("Use full width", value=True)
#     hide_index = st.checkbox("Hide index", value=True)

# # Keep an original copy for diffing
# if "df_original" not in st.session_state:
#     st.session_state.df_original = df_src.copy()

# st.write("### 1) Inline Editing with Column Types, Validation & Widgets")
# st.caption(
#     "Try editing cells below. Status is a selectbox, Priority a bounded number, "
#     "Active a checkbox, Due Date a date picker, and Docs a clickable link. "
#     "The **id** column is read-only."
# )

# # -----------------------------
# # Column configuration
# # -----------------------------
# column_config = {
#     "id": st.column_config.NumberColumn("ID", help="Primary key (read-only).", disabled=True),
#     "name": st.column_config.TextColumn(
#         "Name",
#         help="Free text; min length 2.",
#         validate="^.{2,}$",
#         required=True,
#         max_chars=50,
#         width="medium",
#     ),
#     "owner": st.column_config.TextColumn("Owner", help="Person responsible."),
#     "status": st.column_config.SelectboxColumn(
#         "Status", options=["todo", "doing", "done"], help="Workflow state."
#     ),
#     "priority": st.column_config.NumberColumn(
#         "Priority", min_value=1, max_value=5, step=1, help="1 (highest) → 5 (lowest)"
#     ),
#     "active": st.column_config.CheckboxColumn("Active"),
#     "due_date": st.column_config.DateColumn("Due Date"),
#     "docs": st.column_config.LinkColumn("Docs Link", help="Clickable link."),
# }

# # -----------------------------
# # Data Editor
# # -----------------------------
# edited_df = st.data_editor(
#     df_src,
#     num_rows="dynamic" if allow_add else "fixed",
#     column_config=column_config,
#     use_container_width=use_container_width,
#     hide_index=hide_index,
#     key="data_editor",
# )

# # -----------------------------
# # Selection API
# # -----------------------------
# st.write("### 2) Row Selection & Bulk Actions")
# sel = st.session_state.data_editor.get("selected_rows", []) if isinstance(
#     st.session_state.data_editor, dict
# ) else []
# st.caption("Click the left-most area to select rows; then try bulk actions.")
# colA, colB, colC = st.columns(3)
# with colA:
#     st.metric("Selected rows", len(sel))
# with colB:
#     if st.button("Deactivate selected"):
#         if sel:
#             edited_df.loc[sel, "active"] = False
#             st.success(f"Deactivated {len(sel)} row(s). Scroll up to see changes.")
#         else:
#             st.info("Select one or more rows first.")
# with colC:
#     if st.button("Mark selected as Done"):
#         if sel:
#             edited_df.loc[sel, "status"] = "done"
#             st.success(f"Updated status for {len(sel)} row(s).")
#         else:
#             st.info("Select one or more rows first.")

# # -----------------------------
# # Simple validation feedback
# # -----------------------------
# st.write("### 3) Basic Validation (example)")
# bad_names = edited_df["name"].isna() | (edited_df["name"].astype(str).str.len() < 2)
# bad_priority = ~edited_df["priority"].between(1, 5)

# errs = []
# if bad_names.any():
#     errs.append(f"• {bad_names.sum()} row(s) have invalid **name** (min length 2).")
# if bad_priority.any():
#     errs.append(f"• {bad_priority.sum()} row(s) have **priority** outside 1–5.")

# if errs:
#     st.error("Please fix the following before committing:\n\n" + "\n".join(errs))
# else:
#     st.success("All validation checks passed.")

# # -----------------------------
# # Diff against original
# # -----------------------------
# st.write("### 4) Diff Viewer (Original vs Edited)")
# orig = st.session_state.df_original
# # Align on index shape (naive align; for production consider stable row IDs)
# orig_aligned = orig.reindex(range(len(edited_df))).reset_index(drop=True)
# edited_aligned = edited_df.reset_index(drop=True)

# diff_mask = (orig_aligned.fillna(pd.NA) != edited_aligned.fillna(pd.NA))
# changed_cells = diff_mask.sum().sum()

# with st.expander(f"Show cell-level changes ({changed_cells} changed)"):
#     # Build a human-readable list of changes
#     changes = []
#     for r in range(len(edited_aligned)):
#         for c in edited_aligned.columns:
#             if r < len(orig_aligned) and diff_mask.at[r, c]:
#                 changes.append(
#                     f"Row {r+1}, **{c}**: '{orig_aligned.at[r, c]}' → '{edited_aligned.at[r, c]}'"
#                 )
#     if changes:
#         st.markdown("\n".join([f"- {line}" for line in changes]))
#     else:
#         st.caption("No changes vs original.")

# # -----------------------------
# # Commit / Export
# # -----------------------------
# st.write("### 5) Commit & Export")
# commit_col, export_col = st.columns(2)
# with commit_col:
#     if st.button("✅ Commit edits to session"):
#         st.session_state.df_original = edited_df.copy()
#         st.success("Edits committed. Diff is now reset against the new baseline.")

# with export_col:
#     csv = edited_df.to_csv(index=False).encode("utf-8")
#     st.download_button("⬇️ Download CSV", csv, "edited_table.csv", "text/csv")

# st.caption(
#     "Tips: Use **num_rows='dynamic'** to let users add/remove rows, "
#     "configure per-column widgets with **st.column_config***, and access selection via the editor’s **key**."
# )

# # Footer
# st.markdown("---")
# st.caption(
#     "Built with Streamlit `st.data_editor`. If you see parameter issues, update Streamlit to the latest version."
# )


st.markdown("— or —")

with st.container():
    st.markdown("#### Fetch schema from BigQuery")
    c1, c2, c3 = st.columns([1.3, 1.1, 1.2])
    with c1:
        project_id = st.text_input("Project ID", placeholder="my-gcp-project")
    with c2:
        dataset_id = st.text_input("Dataset", placeholder="analytics")
    with c3:
        table_name = st.text_input("Table", placeholder="events")

    def bq_ident(v: str) -> str:
        return v.strip()

    if project_id and dataset_id and table_name:
        sql = f"""
-- Copy & run in BigQuery to list the table schema
SELECT
  column_name,
  data_type,
  is_nullable,
  ordinal_position
FROM `{bq_ident(project_id)}.{bq_ident(dataset_id)}.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = '{bq_ident(table_name)}'
ORDER BY ordinal_position;
""".strip()
        st.code(sql, language="sql")
        st.caption("Tip: The code block has a copy button on the top-right.")

    with st.expander("Need a ready-made example?"):
        example_sql = """
-- Example: Citi Bike trips schema
SELECT
  column_name,
  data_type,
  is_nullable,
  ordinal_position
FROM `bigquery-public-data.new_york.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'citibike_trips'
ORDER BY ordinal_position;
""".strip()
        st.code(example_sql, language="sql")