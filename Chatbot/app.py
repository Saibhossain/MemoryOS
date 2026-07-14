"""
Chatbot/app.py

Full chatbot UI:
  - Secure Login/Registration for user-specific data isolation
  - Sidebar model selection querying local Ollama instance
  - Scoped profiles and chat session lists
  - Inline message editing for any user message in the thread
  - Clean popover-based image upload toolbar
  - Mobile-responsive layout adjustments
"""
import sys
import os
import urllib.request
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

from DB.users import (
    init_users_table,
    create_user,
    verify_user,
)
from DB.profiles import (
    init_profiles_table,
    create_profile,
    list_profiles,
    rename_profile,
    delete_profile,
    get_profile_name,
    ensure_user_has_profile,
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
from Chatbot.graph import build_graph, describe_image
from Chatbot.utils import build_human_message, message_text, message_has_image_note
from langchain_core.messages import RemoveMessage

st.set_page_config(page_title="MemoryOS Chatbot", layout="wide")

# ---------------------------------------------------------------------
# One-time setup (idempotent)
# ---------------------------------------------------------------------
init_chat_sessions_table()
init_summary_context_table()
init_profiles_table()
init_users_table()

# Helper to fetch local models from Ollama API
def list_local_ollama_models():
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as response:
            data = json.loads(response.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []

# Custom CSS for Mobile Optimization & Aesthetics
st.markdown(
    """
    <style>
    /* Styling adjustments for mobile screens */
    @media (max-width: 768px) {
        .stChatInput {
            padding: 0.5rem !important;
        }
        .stChatMessage {
            padding: 0.5rem !important;
            margin: 0.2rem 0 !important;
        }
        .stButton button {
            width: 100% !important;
            min-height: 44px !important;
        }
        .stTextInput input, .stTextArea textarea {
            font-size: 16px !important;
        }
    }
    
    /* Elegant design for login container */
    .login-box {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 2.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        max-width: 480px;
        margin: 2rem auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()

# ---------------------------------------------------------------------
# Authentication gate
# ---------------------------------------------------------------------
if "username" not in st.session_state:
    st.session_state.username = None

if st.session_state.username is None:
    st.title("🧠 MemoryOS Chatbot")
    st.markdown("Please sign in or register to protect your chat memory.")
    
    tab1, tab2 = st.tabs(["🔒 Sign In", "📝 Register"])
    
    with tab1:
        with st.form("login_form"):
            user = st.text_input("Username").strip().lower()
            pwd = st.text_input("Password", type="password")
            btn = st.form_submit_button("Log In", use_container_width=True)
            if btn:
                if verify_user(user, pwd):
                    st.session_state.username = user
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
    with tab2:
        with st.form("register_form"):
            new_user = st.text_input("Choose Username").strip().lower()
            new_pwd = st.text_input("Choose Password", type="password")
            btn = st.form_submit_button("Register", use_container_width=True)
            if btn:
                if not new_user or not new_pwd:
                    st.error("Fields cannot be empty.")
                elif create_user(new_user, new_pwd):
                    st.success("Registration successful! You can now sign in.")
                else:
                    st.error("Username already exists.")
                    
    st.stop()

# User is authenticated
username = st.session_state.username

# ---------------------------------------------------------------------
# Session state initialization scoped to user
# ---------------------------------------------------------------------
ensure_user_has_profile(username)

if "active_thread_id" not in st.session_state or st.session_state.active_thread_id is None:
    chats = list_chats(username)
    if chats:
        st.session_state.active_thread_id = chats[0][0]
    else:
        st.session_state.active_thread_id = create_chat(username, "New Chat")

if "renaming_thread_id" not in st.session_state:
    st.session_state.renaming_thread_id = None

if "active_profile_id" not in st.session_state or st.session_state.active_profile_id is None:
    profiles = list_profiles(username)
    st.session_state.active_profile_id = profiles[0][0]

if "renaming_profile_id" not in st.session_state:
    st.session_state.renaming_profile_id = None

if "editing_message_id" not in st.session_state:
    st.session_state.editing_message_id = None

# ---------------------------------------------------------------------
# Sidebar: Model, Profiles, Chats, Logout
# ---------------------------------------------------------------------
with st.sidebar:
    st.subheader(f"👤 Logged in as: {username}")
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.username = None
        st.session_state.active_thread_id = None
        st.session_state.active_profile_id = None
        st.session_state.editing_message_id = None
        st.rerun()
        
    st.divider()

    # Model Selection UI
    st.subheader("🤖 Model Selection")
    local_models = list_local_ollama_models()
    default_model = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = default_model

    if default_model not in local_models:
        local_models.insert(0, default_model)

    local_models.append("Custom...")

    selected_model_option = st.selectbox(
        "Ollama LLM",
        options=local_models,
        index=local_models.index(st.session_state.selected_model) if st.session_state.selected_model in local_models else 0,
        label_visibility="collapsed",
    )

    if selected_model_option == "Custom...":
        custom_model = st.text_input("Custom model name", value=st.session_state.selected_model)
        if custom_model and custom_model != st.session_state.selected_model:
            st.session_state.selected_model = custom_model.strip()
            st.rerun()
    elif selected_model_option != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_option
        st.rerun()

    st.divider()

    st.subheader("🧠 Memory Profile")
    profiles = list_profiles(username)
    profile_labels = {p[0]: p[1] for p in profiles}

    selected_profile = st.selectbox(
        "Active Profile",
        options=list(profile_labels.keys()),
        format_func=lambda pid: profile_labels[pid],
        index=list(profile_labels.keys()).index(st.session_state.active_profile_id)
        if st.session_state.active_profile_id in profile_labels else 0,
        label_visibility="collapsed",
    )
    if selected_profile != st.session_state.active_profile_id:
        st.session_state.active_profile_id = selected_profile
        st.rerun()

    pc1, pc2 = st.columns(2)
    if pc1.button("➕ New", use_container_width=True, key="new_profile_btn"):
        new_pid = create_profile(username, "New Profile")
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
        if st.button("Save profile name", key="save_profile_name", use_container_width=True):
            rename_profile(st.session_state.active_profile_id, new_pname.strip() or "Untitled")
            st.session_state.renaming_profile_id = None
            st.rerun()

    if len(profiles) > 1:
        if st.button("🗑️ Delete this profile", key="delete_profile_btn", use_container_width=True):
            delete_profile(st.session_state.active_profile_id, username)
            remaining = list_profiles(username)
            st.session_state.active_profile_id = remaining[0][0]
            st.rerun()

    st.caption("Long-term memory is shared across chats scoped to this profile.")
    st.divider()

    st.subheader("💬 Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        new_id = create_chat(username, "New Chat")
        st.session_state.active_thread_id = new_id
        st.rerun()

    st.divider()

    for thread_id, title, created_at, updated_at, message_count in list_chats(username):
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
                summary, summarized_count = get_summary_context(thread_id)
                badge = "🧵 " if summary else ""

                cols = st.columns([5, 1, 1])
                if cols[0].button(
                    f"{'🟢 ' if is_active else ''}{badge}{title} · {message_count} msgs",
                    key=f"open_{thread_id}",
                    use_container_width=True,
                ):
                    st.session_state.active_thread_id = thread_id
                    st.session_state.editing_message_id = None
                    st.rerun()
                if cols[1].button("✏️", key=f"edit_{thread_id}"):
                    st.session_state.renaming_thread_id = thread_id
                    st.rerun()
                if cols[2].button("🗑️", key=f"del_{thread_id}"):
                    delete_chat(thread_id)
                    if st.session_state.active_thread_id == thread_id:
                        remaining = list_chats(username)
                        st.session_state.active_thread_id = (
                            remaining[0][0] if remaining else create_chat(username, "New Chat")
                        )
                    st.rerun()

# ---------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------
active_thread_id = st.session_state.active_thread_id
config = {
    "configurable": {
        "thread_id": active_thread_id,
        "profile_id": st.session_state.active_profile_id,
        "model": st.session_state.selected_model,
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

notes = list_notes(st.session_state.active_profile_id)
if notes:
    with st.expander(f"🧠 What I remember about you ({len(notes)} notes)", expanded=False):
        for n in notes:
            ncol1, ncol2 = st.columns([5, 1])
            ncol1.write(f"• {n['text']}")
            if ncol2.button("🗑️", key=f"del_note_{n['key']}"):
                delete_note(st.session_state.active_profile_id, n['key'])
                st.rerun()

# Render chat messages with inline editing
for idx, msg in enumerate(messages):
    if msg.type == "tool":
        continue
    role = "user" if msg.type == "human" else "assistant"
    msg_id = getattr(msg, "id", f"idx_{idx}")
    
    with st.chat_message(role):
        if role == "user" and st.session_state.get("editing_message_id") == msg_id:
            edited_text = st.text_area(
                "Edit your message",
                value=message_text(msg),
                key=f"edit_area_{msg_id}",
            )
            col1, col2 = st.columns(2)
            if col1.button("Save & Regenerate", key=f"save_edit_{msg_id}", use_container_width=True):
                # Remove current message and everything after it from the graph history
                removals = [RemoveMessage(id=m.id) for m in messages[idx:]]
                graph.update_state(config, {"messages": removals})
                
                # Insert the newly edited human message and execute
                new_msg = build_human_message(edited_text, image_description=None)
                with st.spinner("Regenerating..."):
                    graph.invoke({"messages": [new_msg]}, config)
                st.session_state.editing_message_id = None
                
                new_state = graph.get_state(config)
                touch_chat(active_thread_id, len(new_state.values.get("messages", [])))
                st.rerun()
            if col2.button("Cancel", key=f"cancel_edit_{msg_id}", use_container_width=True):
                st.session_state.editing_message_id = None
                st.rerun()
        else:
            st.write(message_text(msg))
            if message_has_image_note(msg):
                st.caption("🖼️ this message includes an analyzed image (stored as text)")
            
            # Inline edit button for human messages
            if role == "user":
                if st.button("✏️ Edit", key=f"edit_btn_{msg_id}"):
                    st.session_state.editing_message_id = msg_id
                    st.rerun()

st.divider()

# ---------------------------------------------------------------------
# Input row: toolbar (popover image upload) + text input
# ---------------------------------------------------------------------
tcol1, tcol2 = st.columns([1, 4])
with tcol1:
    with st.popover("🖼️ Attach Image", use_container_width=True):
        uploaded_image = st.file_uploader(
            "Select image",
            type=["png", "jpg", "jpeg", "webp"],
            key=f"uploader_{active_thread_id}",
            label_visibility="collapsed"
        )
        if uploaded_image:
            st.image(uploaded_image, caption="Preview", use_container_width=True)
with tcol2:
    st.write(f"Active Model: **{st.session_state.selected_model}**")

user_text = st.chat_input("Type your message...")

if user_text is not None:
    image_description = None

    if uploaded_image is not None:
        with st.spinner("👁️ Analyzing image..."):
            try:
                image_bytes = uploaded_image.getvalue()
                mime = uploaded_image.type or "image/png"
                image_description = describe_image(image_bytes, mime, model_name=st.session_state.selected_model)
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
            st.image(uploaded_image, width=200, caption="Uploaded (analyzed only)")
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
                    if not getattr(candidate, "tool_calls", None):
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
                    "🧵 Older messages were just folded into the running summary. See the summary panel above.",
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