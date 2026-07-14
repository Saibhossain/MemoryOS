"""
Chatbot/app.py  (Phase 2, revised)

Full chatbot UI:
  - New chat / delete chat / rename chat (sidebar)
  - Persistent, UNTRIMMED conversation memory via LangGraph + PostgresSaver
  - Image upload: analyzed once via a vision call, only the resulting text
    description is stored - no image bytes ever touch Postgres
  - Automatic, visible summarization for what gets SENT to the model
    (full history always stays visible in the UI and in storage)
  - Edit-last-message + regenerate

Run with:
    streamlit run Chatbot/app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from DB.profiles import (
    init_profiles_table,
    create_profile,
    list_profiles,
    rename_profile,
    delete_profile,
    get_profile_name,
    DEFAULT_PROFILE_ID,
)
from DB.long_term_memory import list_notes, delete_note


from DB.chat_sessions import (
    init_chat_sessions_table,
    create_chat,
    list_chats,
    rename_chat,
    delete_chat,
    touch_chat,
    get_chat_title,
)
from DB.summary_context import init_summary_context_table, get_summary_context
from Chatbot.graph import build_graph, regenerate_from_edit, describe_image
from Chatbot.utils import build_human_message, message_text, message_has_image_note

st.set_page_config(page_title="MemoryOS Chatbot", layout="wide")

# ---------------------------------------------------------------------
# One-time setup (idempotent) + cached, long-lived resources
# ---------------------------------------------------------------------
init_chat_sessions_table()
init_summary_context_table()
init_profiles_table()

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

if "active_profile_id" not in st.session_state:
    st.session_state.active_profile_id = DEFAULT_PROFILE_ID

if "renaming_profile_id" not in st.session_state:
    st.session_state.renaming_profile_id = None

# ---------------------------------------------------------------------
# Sidebar: chat list + new/delete/rename controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.subheader("🧠 Profile")

    profiles = list_profiles()
    profile_labels = {p[0]: p[1] for p in profiles}  # profile_id -> name

    selected_name = st.selectbox(
        "Active profile",
        options=list(profile_labels.keys()),
        format_func=lambda pid: profile_labels[pid],
        index=list(profile_labels.keys()).index(st.session_state.active_profile_id)
        if st.session_state.active_profile_id in profile_labels else 0,
        label_visibility="collapsed",
    )
    if selected_name != st.session_state.active_profile_id:
        st.session_state.active_profile_id = selected_name
        st.rerun()

    pc1, pc2 = st.columns(2)
    if pc1.button("➕ New", use_container_width=True, key="new_profile_btn"):
        new_pid = create_profile("New Profile")
        st.session_state.active_profile_id = new_pid
        st.rerun()
    if pc2.button("✏️ Rename", use_container_width=True, key="rename_profile_btn"):
        st.session_state.renaming_profile_id = st.session_state.active_profile_id
        st.rerun()

    if st.session_state.renaming_profile_id == st.session_state.active_profile_id:
        new_pname = st.text_input(
            "New profile name",
            value=profile_labels.get(st.session_state.active_profile_id, ""),
            key="profile_rename_input",
        )
        if st.button("Save profile name", key="save_profile_name"):
            rename_profile(st.session_state.active_profile_id, new_pname.strip() or "Untitled")
            st.session_state.renaming_profile_id = None
            st.rerun()

    if st.session_state.active_profile_id != DEFAULT_PROFILE_ID:
        if st.button("🗑️ Delete this profile", key="delete_profile_btn"):
            delete_profile(st.session_state.active_profile_id)
            st.session_state.active_profile_id = DEFAULT_PROFILE_ID
            st.rerun()

    st.caption(
        "Long-term memory (facts the agent remembers about you) is scoped "
        "to this profile, not to any single chat — it follows you across "
        "every conversation."
    )
    st.divider()

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
                # Cheap lookup: summary_context is a plain SQL table now,
                # not something requiring graph.get_state() to decode.
                summary, summarized_count = get_summary_context(thread_id)
                badge = "🧵 " if summary else ""

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
        "🧵 = this chat has a running summary. Full conversation history "
        "is always kept - the summary only affects what's sent to the "
        "model, never what's shown in the UI."
    )


# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
active_thread_id = st.session_state.active_thread_id
config = {
    "configurable": {
        "thread_id": active_thread_id,
        "profile_id": st.session_state.active_profile_id,   # NEW
    }
}

st.title(get_chat_title(active_thread_id))

state = graph.get_state(config)
messages = state.values.get("messages", [])

current_summary, summarized_count = get_summary_context(active_thread_id)

if current_summary:
    with st.expander(
        f"📋 Conversation summary (covers first {summarized_count} messages)",
        expanded=False,
    ):
        st.write(current_summary)
        st.caption(
            "The model sees this summary plus only the messages after it - "
            "not the full raw history - to stay within its context window. "
            "Every message below is still fully preserved and shown as-is."
        )

notes = list_notes(st.session_state.active_profile_id)
if notes:
    with st.expander(f"🧠 What I remember about you ({len(notes)} notes)", expanded=False):
        for n in notes:
            ncol1, ncol2 = st.columns([5, 1])
            ncol1.write(f"• {n['text']}")
            if ncol2.button("🗑️", key=f"del_note_{n['key']}"):
                delete_note(st.session_state.active_profile_id, n['key'])
                st.rerun()
        st.caption(
            "These notes were saved automatically by the agent, or when "
            "you explicitly asked it to remember something. They apply to "
            "every chat under this profile, not just this one."
        )

for msg in messages:
    if msg.type == "tool":
        continue   # NEW — tool results are internal, not conversational turns
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.write(message_text(msg))
        if message_has_image_note(msg):
            st.caption("🖼️ this message includes an analyzed image (stored as text)")

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
            new_msg = build_human_message(edited_text, image_description=None)
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
    "Attach an image (optional - it will be analyzed and stored as a "
    "text description, not saved as image data)",
    type=["png", "jpg", "jpeg", "webp"],
    key=f"uploader_{active_thread_id}",
)

user_text = st.chat_input("Type your message...")

if user_text is not None:
    image_description = None

    if uploaded_image is not None:
        with st.spinner("👁️ Analyzing image..."):
            try:
                image_bytes = uploaded_image.getvalue()
                mime = uploaded_image.type or "image/png"
                image_description = describe_image(image_bytes, mime)
            except Exception as e:
                st.warning(
                    f"Image analysis failed ({e}). This usually means "
                    f"OLLAMA_MODEL isn't a vision-capable model. Try "
                    f"`ollama pull qwen2.5vl` and set OLLAMA_MODEL to it. "
                    f"Continuing with text only."
                )

    human_msg = build_human_message(user_text, image_description)

    with st.chat_message("user"):
        st.write(user_text)
        if uploaded_image is not None:
            st.image(uploaded_image, width=200, caption="Uploaded (not stored - analyzed only)")
        if image_description:
            st.caption(f"👁️ Vision analysis: {image_description}")

    with st.chat_message("assistant"):
        status = st.empty()
        response_box = st.empty()
        final_response = None
        summarized_this_turn = False

        try:
            status.markdown("_Thinking..._")
            for chunk in graph.stream(
                {"messages": [human_msg]}, config, stream_mode="updates"
            ):
                if "model" in chunk:
                    candidate = chunk["model"]["messages"][-1]
                    if not getattr(candidate,"tool_calls",None):
                        final_response = candidate 
                    status.markdown("_Thinking..._")
                if "tools" in chunk:                                    
                    status.markdown("🧠 _Checking/updating memory..._")  
                if "summarize" in chunk:
                    summarized_this_turn = True
                    status.markdown("🧵 _Compacting older messages into summary..._")

            status.empty()
            if final_response is not None and message_text(final_response).strip():
                response_box.write(message_text(final_response))
            else:
                response_box.write("_(No text response — check the console/logs; the model may have only called a tool.)_")
                
            if summarized_this_turn:
                st.info(
                    "🧵 Older messages were just folded into the running "
                    "summary to keep the model's input within its context "
                    "window. Nothing was deleted - the full conversation "
                    "above is still complete. See the summary panel above.",
                    icon="🧵",
                )
        except Exception as e:
            status.empty()
            st.error(f"Model call failed: {e}")

    if len(messages) == 0:
        auto_title = user_text.strip()[:40] or "New Chat"
        rename_chat(active_thread_id, auto_title)

    new_state = graph.get_state(config)
    touch_chat(active_thread_id, len(new_state.values.get("messages", [])))
    st.rerun()