"""
F.R.I.D.A.Y - AI-Powered Analytics Assistant

Main application entry point for the refactored Streamlit app.
"""

import streamlit as st

# Core configuration and state
from src.config import settings
from src.ui.state import init_session_state
from src.context.persistence import load_persisted_context_to_session

# Import UI components
from src.ui.components.sidebar import render_sidebar
from src.ui.pages.context_page import render_ctx_page
from src.ui.pages.view_context import render_view_context_page
from src.ui.pages.analysis import render_analysis_page

# Set page configuration
st.set_page_config(
    page_title=settings.APP_NAME,
    page_icon=settings.APP_ICON,
   layout="centered",
)

def main():
    # Initialize session state
    init_session_state()

    # Load persisted context (if any)
    load_persisted_context_to_session()

    # Render Sidebar
    render_sidebar()

    # Route to appropriate page
    if st.session_state.view == "context":
        render_ctx_page()
    elif st.session_state.view == "view_context":
        render_view_context_page()
    else:
        # Default view
        st.session_state.view = "analysis"
        render_analysis_page()

if __name__ == "__main__":
    main()
