"""
DB/chat_sessions.py

LangGraph's PostgresSaver persists conversation *state* (the messages)
keyed by thread_id, but it has no concept of a "chat title" or "list all
my conversations" - that is an application-level concept, not a LangGraph
one. This module owns a small `chat_sessions` table that sits next to
LangGraph's own tables (checkpoints, checkpoint_blobs, checkpoint_writes)
and gives the Streamlit sidebar something to list, rename, and delete.
"""
import uuid

from DB.connection import get_pool


def init_chat_sessions_table():
    """Idempotent - safe to call on every app startup."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                thread_id      TEXT PRIMARY KEY,
                title          TEXT NOT NULL DEFAULT 'New Chat',
                created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                message_count  INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at "
            "ON chat_sessions (updated_at DESC);"
        )


def create_chat(title: str = "New Chat") -> str:
    thread_id = str(uuid.uuid4())
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (thread_id, title) VALUES (%s, %s)",
            (thread_id, title),
        )
    return thread_id


def list_chats():
    """Returns rows: (thread_id, title, created_at, updated_at, message_count)"""
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT thread_id, title, created_at, updated_at, message_count
            FROM chat_sessions
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return rows


def get_chat_title(thread_id: str) -> str:
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT title FROM chat_sessions WHERE thread_id = %s", (thread_id,)
        ).fetchone()
    return row[0] if row else "New Chat"


def rename_chat(thread_id: str, new_title: str):
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "UPDATE chat_sessions SET title = %s, updated_at = now() WHERE thread_id = %s",
            (new_title, thread_id),
        )


def touch_chat(thread_id: str, message_count: int):
    """Called after every model turn so the sidebar shows recency + size."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            UPDATE chat_sessions
            SET updated_at = now(), message_count = %s
            WHERE thread_id = %s
            """,
            (message_count, thread_id),
        )


def delete_chat(thread_id: str):
    """
    Deletes the metadata row AND the underlying LangGraph checkpoint data
    for this thread. PostgresSaver does not cascade-delete automatically,
    so we clean up its three tables by hand.
    """
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
        conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
        conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
        conn.execute("DELETE FROM chat_sessions WHERE thread_id = %s", (thread_id,))