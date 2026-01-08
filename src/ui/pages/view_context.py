"""View context page"""

import streamlit as st
import pandas as pd

def render_view_context_page():
    """Displays the currently set context (table and schema)."""
    st.header("View Current Context")

    if not st.session_state.ctx_set or not st.session_state.selected_tables:
        st.error("No context is currently set.")
        if st.button("Set Context", icon="🧠", use_container_width=True):
            st.session_state.view = "context"
            st.rerun()
        return

    st.write(f"**Total Tables:** {len(st.session_state.selected_tables)}")
    
    for idx, table_info in enumerate(st.session_state.selected_tables):
        table_fqn = f"{table_info['project']}.{table_info['dataset']}.{table_info['table']}"
        with st.expander(f"📄 {idx+1}. {table_fqn}", expanded=(idx==0)):
            st.markdown(f"**Description:** {table_info.get('description') or 'No description'}")
            
            # Get schema for this table
            schema_df = st.session_state.schemas.get(table_fqn, pd.DataFrame())
            
            # Fallback for legacy
            if schema_df.empty and idx == 0 and not st.session_state.schema_df.empty:
                schema_df = st.session_state.schema_df

            if schema_df.empty:
                 st.warning("Schema data is missing for this table.")
            else:
                st.dataframe(
                    schema_df, 
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "column_description": st.column_config.TextColumn("Column Description")
                    }
                )
    
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Return to Analysis", icon="⬅️", use_container_width=True):
            st.session_state.view = "analysis"
            st.rerun()
    with c2:
        if st.button("Change Context", icon="🔄", use_container_width=True):
            st.session_state.view = "context"
            st.rerun()
