import os
import sys
import uuid
import streamlit as st

# Ensure proper path routing from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import AIMessage, HumanMessage
from Chatbot.graph import app
from Chatbot.utils import build_human_message, message_has_image, message_text
from DB.connection import get_pool, get_conn_string
from langgraph.checkpoint.postgres import PostgresSaver

st.set_page_config(page_title="MemoryOS - Chat Interface", layout="wide", page_icon="🧠")
pool = get_pool()

# ---------------------------------------------------------------------
# Session Metadata Synchronization Helpers
# ---------------------------------------------------------------------
def sync_chat_session(thread_id: str, title: str, message_count: int):
    """Upserts tracking logs to the chat_sessions dashboard index table."""
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_sessions (thread_id, title, message_count, updated_at)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (thread_id) DO UPDATE SET
                    message_count = EXCLUDED.message_count,
                    updated_at = CURRENT_TIMESTAMP;
            """, (thread_id, title[:50], message_count))

def purge_chat_session(thread_id: str):
    """Removes records from application index and systemic checkpointer tables."""
    # 1. Clear application session indexes
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_sessions WHERE thread_id = %s;", (thread_id,))
            
    # 2. Clear core LangGraph internal snapshots for this thread execution
    with PostgresSaver.from_conn_string(get_conn_string()) as cp:
        cp.delete_thread(thread_id)

def fetch_all_sessions():
    """Retrieves list of existing active sessions for side panel navigation."""
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT thread_id, title FROM chat_sessions ORDER BY updated_at DESC;")
                return cur.fetchall()
    except Exception:
        return []

# ---------------------------------------------------------------------
# Sidebar Panel Layout
# ---------------------------------------------------------------------
st.sidebar.title("🧠 Agentic Memory OS")
st.sidebar.subheader("Phase 1: Short-Term Thread State")

# Select/Load Thread Sessions
existing_chats = fetch_all_sessions()
chat_options = {f"✨ {title} ({tid[:6]})": tid for tid, title in existing_chats}

if "thread_id" not in st.session_state:
    if chat_options:
        st.session_state.thread_id = list(chat_options.values())[0]
    else:
        st.session_state.thread_id = str(uuid.uuid4())

# Session Operations
col_new, col_del = st.sidebar.columns(2)
with col_new:
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

with col_del:
    if st.button("🗑️ Delete Chat", use_container_width=True):
        purge_chat_session(st.session_state.thread_id)
        st.session_state.thread_id = str(uuid.uuid4())
        st.sidebar.success("Database records dropped.")
        st.rerun()

# Workspace Selector Dropdown
if chat_options:
    selected_name = st.sidebar.selectbox(
        "Active Conversations",
        options=list(chat_options.keys()),
        index=list(chat_options.values()).index(st.session_state.thread_id) if st.session_state.thread_id in chat_options.values() else 0
    )
    if chat_options[selected_name] != st.session_state.thread_id:
        st.session_state.thread_id = chat_options[selected_name]
        st.rerun()

st.sidebar.divider()
uploaded_image = st.sidebar.file_uploader("Multimodal Vision Input", type=["png", "jpg", "jpeg"])

# ---------------------------------------------------------------------
# Thread State Processing Engine
# ---------------------------------------------------------------------
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# Fetch the existing persistent message stack from PostgreSQL
try:
    current_state = app.get_state(config)
    messages = current_state.values.get("messages", [])
except Exception:
    messages = []

st.title("🗪 Local Agent Core")
st.caption(f"Connected to thread checkpoint tracking: `{st.session_state.thread_id}`")

# Render historical conversation elements
for idx, msg in enumerate(messages):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(message_text(msg))
            if message_has_image(msg):
                st.caption("🖼️ *Attached Multimodal Context Payload stored in `checkpoint_blobs`*")
            
            # Message State Editing Engine
            with st.expander("✏️ Edit this checkpoint position"):
                edit_input = st.text_input("Modify text prompt:", value=message_text(msg), key=f"inp_{msg.id or idx}")
                if st.button("Fork State Flow Here", key=f"btn_{msg.id or idx}"):
                    # Write modified payload directly targeting historical ID node pathing
                    updated_msg = HumanMessage(content=edit_input, id=msg.id)
                    app.update_state(config, {"messages": [updated_msg]}, as_node="chatbot")
                    st.success("Graph execution path shifted successfully.")
                    st.rerun()
                    
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Process incoming stream events
if prompt_input := st.chat_input("Enter message sequence context..."):
    with st.chat_message("user"):
        st.write(prompt_input)
        
    # Standardize structure via utilities wrapper
    formatted_msg = build_human_message(prompt_input, uploaded_image)
    
    with st.chat_message("assistant"):
        with st.spinner("Invoking local graph computation nodes..."):
            # Execute pipeline and stream tracking states downstream inside postgres
            output = app.invoke({"messages": [formatted_msg]}, config=config)
            ai_response = output["messages"][-1].content
            st.write(ai_response)
            
    # Calculate metadata details to refresh the monitoring application layout indexes
    updated_messages = output.get("messages", [])
    inferred_title = prompt_input if len(prompt_input) < 45 else f"{prompt_input[:42]}..."
    if len(updated_messages) > 2:
        # Keep title static based on initial prompt context if it exists
        try:
            with pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT title FROM chat_sessions WHERE thread_id = %s;", (st.session_state.thread_id,))
                    res = cur.fetchone()
                    if res: inferred_title = res[0]
        except Exception:
            pass
            
    sync_chat_session(st.session_state.thread_id, inferred_title, len(updated_messages))
    st.rerun()