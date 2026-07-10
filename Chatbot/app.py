"""
Chatbot/app.py

Full chatbot UI:
  - New chat / delete chat / rename chat (sidebar)
  - Persistent conversation memory via LangGraph + PostgresSaver
  - Image upload (multimodal message, if your Ollama model supports vision)
  - Edit-last-message + regenerate (rewinds and re-invokes the graph)

Run with:
    streamlit run Chatbot/app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from DB.chat_sessions import (
    init_chat_sessions_table,
    create_chat,
    list_chats,
    rename_chat,
    delete_chat,
    touch_chat,
    get_chat_title,
)
from Chatbot.graph import build_graph, regenerate_from_edit
from Chatbot.utils import build_human_message, message_text, message_has_image

st.set_page_config(page_title="MemoryOS Chatbot", layout="wide")

# ---------------------------------------------------------------------
# One-time setup (idempotent) + cached, long-lived resources
# ---------------------------------------------------------------------
init_chat_sessions_table()


@st.cache_resource
def get_graph():
    """Built once per Streamlit server process, not once per rerun -
    otherwise we'd open a new Postgres connection on every button click."""
    return build_graph()


graph = get_graph()

# ---------------------------------------------------------------------
# Session state: which chat is currently open
# ---------------------------------------------------------------------
if "active_thread_id" not in st.session_state:
    chats = list_chats()
    if chats:
        st.session_state.active_thread_id = chats[0][0]
    else:
        st.session_state.active_thread_id = create_chat("New Chat")

if "renaming_thread_id" not in st.session_state:
    st.session_state.renaming_thread_id = None

if "editing_last_message" not in st.session_state:
    st.session_state.editing_last_message = False


# ---------------------------------------------------------------------
# Sidebar: chat list + new/delete/rename controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_chat("New Chat")
        st.session_state.active_thread_id = new_id
        st.session_state.editing_last_message = False
        st.rerun()

    st.divider()

    for thread_id, title, created_at, updated_at, message_count in list_chats():
        is_active = thread_id == st.session_state.active_thread_id
        row = st.container()

        with row:
            if st.session_state.renaming_thread_id == thread_id:
                new_title = st.text_input(
                    "Rename", value=title, key=f"rename_input_{thread_id}",
                    label_visibility="collapsed",
                )
                c1, c2 = st.columns(2)
                if c1.button("Save", key=f"save_{thread_id}", use_container_width=True):
                    rename_chat(thread_id, new_title.strip() or "Untitled")
                    st.session_state.renaming_thread_id = None
                    st.rerun()
                if c2.button("Cancel", key=f"cancel_{thread_id}", use_container_width=True):
                    st.session_state.renaming_thread_id = None
                    st.rerun()
            else:
                cols = st.columns([5, 1, 1])
                if cols[0].button(
                    f"{'🟢 ' if is_active else ''}{title}  ·  {message_count} msgs",
                    key=f"open_{thread_id}",
                    use_container_width=True,
                ):
                    st.session_state.active_thread_id = thread_id
                    st.session_state.editing_last_message = False
                    st.rerun()
                if cols[1].button("✏️", key=f"edit_{thread_id}"):
                    st.session_state.renaming_thread_id = thread_id
                    st.rerun()
                if cols[2].button("🗑️", key=f"del_{thread_id}"):
                    delete_chat(thread_id)
                    if st.session_state.active_thread_id == thread_id:
                        remaining = list_chats()
                        st.session_state.active_thread_id = (
                            remaining[0][0] if remaining else create_chat("New Chat")
                        )
                    st.rerun()

    st.divider()
    st.caption(
        "Memory is stored in Postgres via LangGraph's PostgresSaver, "
        "keyed by thread_id. Open the DB monitor app to watch it live."
    )


# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
active_thread_id = st.session_state.active_thread_id
config = {"configurable": {"thread_id": active_thread_id}}

st.title(get_chat_title(active_thread_id))

state = graph.get_state(config)
messages = state.values.get("messages", [])

for i, msg in enumerate(messages):
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message_text(msg))
        if message_has_image(msg):
            st.caption("🖼️ image attached")

# Edit-last-message control (only offered if there's at least one human turn)
last_human_msg = next((m for m in reversed(messages) if m.type == "human"), None)

if last_human_msg is not None:
    with st.expander("✏️ Edit last message & regenerate"):
        edited_text = st.text_area(
            "Edit your last message",
            value=message_text(last_human_msg),
            key="edit_box",
        )
        if st.button("Regenerate response"):
            new_msg = build_human_message(edited_text, uploaded_image=None)
            with st.spinner("Regenerating..."):
                regenerate_from_edit(graph, config, new_msg)
            new_state = graph.get_state(config)
            touch_chat(active_thread_id, len(new_state.values.get("messages", [])))
            st.rerun()

st.divider()

# ---------------------------------------------------------------------
# Input row: text + optional image
# ---------------------------------------------------------------------
uploaded_image = st.file_uploader(
    "Attach an image (optional - only used if your Ollama model supports vision)",
    type=["png", "jpg", "jpeg", "webp"],
    key=f"uploader_{active_thread_id}",
)

user_text = st.chat_input("Type your message...")

if user_text is not None:
    human_msg = build_human_message(user_text, uploaded_image)

    with st.chat_message("user"):
        st.write(user_text)
        if uploaded_image is not None:
            st.image(uploaded_image, width=200)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = graph.invoke({"messages": [human_msg]}, config)
                reply = result["messages"][-1]
                st.write(message_text(reply))
            except Exception as e:
                st.error(
                    f"Model call failed: {e}\n\n"
                    "If you attached an image, your Ollama model may not "
                    "support vision. Try a vision model such as "
                    "`qwen2.5vl` or `llava`, or resend without the image."
                )

    # If this was the first message, auto-title the chat from it
    if len(messages) == 0:
        auto_title = user_text.strip()[:40] or "New Chat"
        rename_chat(active_thread_id, auto_title)

    new_state = graph.get_state(config)
    touch_chat(active_thread_id, len(new_state.values.get("messages", [])))
    st.rerun()