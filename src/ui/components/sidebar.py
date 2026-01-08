"""Sidebar component"""

import streamlit as st
from src.context.manager import clear_context

def render_sidebar():
    """Renders the application sidebar."""
    st.sidebar.title("F.R.I.D.A.Y")
    st.sidebar.caption("AI-Powered Analytics Assistant")
    
    st.sidebar.title("Context")
    if not st.session_state.ctx_set:
        st.sidebar.warning("Context is empty", icon="⚠️")
        if st.sidebar.button("Set Context", icon="🧠", use_container_width=True):
            st.session_state.view = "context"
            st.rerun()
    else:
        st.sidebar.success("Context set", icon="✅")
        if st.session_state.selected_tables:
            count = len(st.session_state.selected_tables)
            st.sidebar.write(f"**Tables:** {count} table(s) in context")
            for row in st.session_state.selected_tables:
                 st.sidebar.caption(f"• `{row['table']}`")
        
        if st.sidebar.button("View Context Details", icon="📄", use_container_width=True):
            st.session_state.view = "view_context"
            st.rerun()
        
        if st.sidebar.button("Change Context", icon="🔄", use_container_width=True):
            st.session_state.view = "context"
            st.rerun()
    
    with st.sidebar.expander("ℹ️ How it works", expanded=True):
        st.markdown("""
1. **Authenticate** with Google to enable access to your data warehouse.
2. **Set context**: choose a **single table** via schema upload or the BQ picker.
3. **Ask**: Describe what you want to analyze. The AI will propose a plan.
4. **Refine**: Chat with the AI to edit the plan until you're happy.
5. **Approve**: Click "Approve Plan" to generate the final SQL query.
6. **Run**: Execute the generated SQL against BigQuery and see the results.
7. **Visualize**: If applicable, click "Create Chart" to see a visual representation.
""")
