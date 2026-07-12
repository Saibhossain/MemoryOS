"""
Chatbot/app.py  (Phase 2)

Adds on top of Phase 1:
  - Streaming graph execution so we can show a distinct "🧵 Compacting..."
    spinner state when the summarize_node actually runs, instead of the
    generic "Thinking..." spinner
  - A collapsible "📋 Conversation summary" panel showing the current
    running summary for the active thread (if one exists)
  - A sidebar badge showing effective message count + whether a chat has
    been summarized at least once

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


# ---------------------------------------------------------------------
# Sidebar: chat list + new/delete/rename controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_chat("New Chat")
        st.session_state.active_thread_id = new_id
        st.rerun()

    st.divider()

    for thread_id, title, created_at, updated_at, message_count in list_chats():
        is_active = thread_id == st.session_state.active_thread_id

        with st.container():
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
                # Peek at whether this thread has an active summary, for
                # the 🧵 badge - cheap since get_state just reads Postgres.
                thread_config = {"configurable": {"thread_id": thread_id}}
                thread_state = graph.get_state(thread_config)
                has_summary = bool(thread_state.values.get("summary"))
                badge = "🧵 " if has_summary else ""

                cols = st.columns([5, 1, 1])
                if cols[0].button(
                    f"{'🟢 ' if is_active else ''}{badge}{title}  ·  {message_count} msgs",
                    key=f"open_{thread_id}",
                    use_container_width=True,
                ):
                    st.session_state.active_thread_id = thread_id
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
        "🧵 = this chat has been summarized at least once. "
        "Message count shown is the *effective* count sent to the model "
        "(after trimming), not the full historical count."
    )


# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
active_thread_id = st.session_state.active_thread_id
config = {"configurable": {"thread_id": active_thread_id}}

st.title(get_chat_title(active_thread_id))

state = graph.get_state(config)
messages = state.values.get("messages", [])
current_summary = state.values.get("summary", "")

if current_summary:
    with st.expander("📋 Conversation summary (older messages condensed)", expanded=False):
        st.write(current_summary)
        st.caption(
            "The model sees this summary plus the recent messages below - "
            "not the full raw history, to stay within its context window."
        )

for msg in messages:
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
        status = st.empty()
        response_box = st.empty()
        final_response = None
        summarized_this_turn = False

        try:
            status.markdown("_Thinking..._")
            # stream_mode="updates" yields one chunk per node as it
            # finishes, keyed by node name - this is what lets us show
            # a different spinner state when summarize_node runs, since
            # both nodes execute inside the same graph.invoke-equivalent
            # call rather than as separate round trips.
            for chunk in graph.stream(
                {"messages": [human_msg]}, config, stream_mode="updates"
            ):
                if "model" in chunk:
                    final_response = chunk["model"]["messages"][-1]
                    status.markdown("_Thinking..._")
                if "summarize" in chunk:
                    summarized_this_turn = True
                    status.markdown("🧵 _Compacting older messages into summary..._")

            status.empty()
            if final_response is not None:
                response_box.write(message_text(final_response))
            if summarized_this_turn:
                st.info(
                    "🧵 Older messages were just condensed into the running "
                    "summary to keep the conversation within the model's "
                    "context window. See the summary panel above.",
                    icon="🧵",
                )
        except Exception as e:
            status.empty()
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