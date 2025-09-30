import streamlit as st

PAGE_TITLE = "GenSQL"

# --- Page Configuration ---
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="✨",
)

st.markdown("<h2 style='text-align: center;'>What are you analyzing today?</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Start generating SQLs with A.I. for your analyses</p>", unsafe_allow_html=True)

st.sidebar.header("Memory")

st.sidebar.header("Projects")

# Ensure messages state exists
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

tab1, tab2, tab3 = st.tabs(["Step 1. Name the project",
                             "Step 2. Select database table", 
                             "Step 3. Start generating SQLs!"]
                           )

with tab1:
    st.markdown("### Name your project")
    project_name = st.text_input(
        "Project Name",
        placeholder="E.g. Q4 sales analysis",
        label_visibility="collapsed",  # hide label to make it look cleaner
        key="project_name"
    )
    if project_name:
        st.success(f"Project '{project_name}' created! Proceed to the next step.")
    else:
        st.info("Please enter a project name to proceed.")

with tab2:
    prj_name = project_name if project_name else "<Please name your project first in Step 1>"
    st.markdown(f"<h2 style='text-align: center;'>{prj_name}</h2>", unsafe_allow_html=True)
    st.markdown("### Select your database table")
    if st.button("Authenticate with Google"):
        st.success("Authenticated successfully! Now select your database table.")
        # todo: list tables from GCP BigQuery
        if st.button("Set table"):
            st.success("Table selected! Proceed to the next step.")

with tab3:
    prj_name = project_name if project_name else "<Please name your project first in Step 1>"
    st.markdown(f"<h2 style='text-align: center;'>{prj_name}</h2>", unsafe_allow_html=True)
    st.markdown("### Start generating SQLs!")
    st.info("Describe your analysis requirement and let A.I. generate the SQL for you.")
    
    # Ensure messages state exists
    if "messages" not in st.session_state:
        st.session_state.messages = []


    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    
    col1, col2 = st.columns([5, 1])
    with col1:
            user_input = st.text_input(
            "Type your requirement and generate the SQL",
            placeholder="Show me the total sales lost by department for the last quarter.",
            label_visibility="collapsed",  # hide label to make it look cleaner
            key="chat_input"
        )
            
    with col2:
        send_button = st.button("Send", use_container_width=True)

    if send_button and st.session_state.chat_input:
        st.session_state.messages.append({"role": "user", "content": st.session_state.chat_input})
        # Placeholder for AI response
        ai_response = "This is a placeholder response. (Integrate your AI model here.)"
        st.session_state.messages.append({"role": "ai", "content": ai_response})
        # st.session_state.chat_input = ""  # Clear the input after sending

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**GenSQL:** {msg['content']}")

# # Place text input and button on the same row using st.columns with tighter spacing
# col1, col2 = st.columns([5, 1])

# with col1:
#     user_input = st.text_input(
#         "Type your requirement and generate the SQL",
#         placeholder="Show me the total sales lost by department for the last quarter.",
#         label_visibility="collapsed",  # hide label to make it look cleaner
#         key="chat_input"
#     )

# with col2:
#     send_button = st.button("Send", use_container_width=True)

# if send_button and st.session_state.chat_input:
#     st.session_state.messages.append({"role": "user", "content": st.session_state.chat_input})
#     # Placeholder for AI response
#     ai_response = "This is a placeholder response. (Integrate your AI model here.)"
#     st.session_state.messages.append({"role": "ai", "content": ai_response})
#     st.session_state.chat_input = ""  # Clear the input after sending

# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.markdown(f"**You:** {msg['content']}")
#     else:
#         st.markdown(f"**GenSQL:** {msg['content']}")
