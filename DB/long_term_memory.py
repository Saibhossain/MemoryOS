"""
DB/long_term_memory.py

Thin wrapper around LangGraph's PostgresStore - the cross-thread,
long-term memory mechanism. Unlike checkpoints (thread-scoped), the store
is keyed by (namespace, key) and is meant to be read/written explicitly,
not automatically on every graph step.

Design for this project (per the Phase 3 plan): freeform timestamped
notes, no fixed schema/keys. Each note is stored under:

    namespace = (profile_id, "notes")
    key       = uuid4 (one per note)
    value     = {"text": ..., "created_at": ..., "source_thread_id": ...}

Reading happens by pulling ALL notes for a profile (list_notes) and
injecting them as context every turn - not a per-turn tool call. Writing
IS a tool call, made by the model via Chatbot/memory_tools.py.
"""

import uuid
from datetime import datetime, timezone
from psycopg import Connection

from langgraph.store.postgres import PostgresStore
from DB.connection import get_conn_string
_store: PostgresStore | None = None

def get_store() -> PostgresStore:
    """
    Same pattern as Chatbot/graph.py's checkpointer: open a plain psycopg
    connection ourselves and keep it alive for the app's lifetime, rather
    than using PostgresStore.from_conn_string(...) as a `with` block (which
    would close the connection immediately - wrong for a long-running
    Streamlit app). Cached at module level so every caller in the process
    shares one store/connection instead of opening a new one per call.
    """
    global _store
    if _store is None:
        conn = Connection.connect(get_conn_string(), autocommit=True)
        _store = PostgresStore(conn)
    return _store

def add_note(profile_id: str, text: str, source_thread_id: str | None = None) -> str:
    """Writes one freeform note under a profile's namespace. Returns the
    note's key (a uuid4) so callers can reference it (e.g. for deletion)."""
    store = get_store()
    namespace = (profile_id, "notes")
    note_id = str(uuid.uuid4())

    value = {
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_thread_id": source_thread_id,
    }
    store.put(namespace, note_id, value)
    return note_id

def list_notes(profile_id: str) -> list[dict]:
    """
    Returns all notes for a profile, oldest first, each as:
        {"key": ..., "text": ..., "created_at": ..., "source_thread_id": ...}

    Used both to inject context into the model every turn and to render
    the "What I remember" panel in the UI.
    """
    store = get_store()
    namespace = (profile_id, "notes")
    items = store.search(namespace)

    notes = []
    for item in items:
        notes.append(
            {
                "key": item.key,
                "text": item.value.get("text", ""),
                "created_at": item.value.get("created_at", ""),
                "source_thread_id": item.value.get("source_thread_id"),
            }
        )
    notes.sort(key=lambda n: n["created_at"])
    return notes

def delete_note(profile_id: str, note_id: str):
    store = get_store()
    namespace = (profile_id, "notes")
    store.delete(namespace, note_id)

def notes_as_context(profile_id: str) -> str:
    """
    Formats all notes for a profile into a block of text suitable for
    injection as a SystemMessage. Returns an empty string if there are no
    notes yet, so callers can skip adding an empty system message.
    """
    notes = list_notes(profile_id)
    if not notes:
        return ""

    lines = [f"- {n['text']}" for n in notes]
    return "Long-term memory - things you know about this user from past conversations:\n" + "\n".join(lines)
