"""
DB/summary_context.py

Stores the running conversation summary *outside* of LangGraph's
checkpointed message state. This is the key fix to a bug in the first
Phase 2 pass: summarizing used to delete old messages from state via
RemoveMessage, which meant the full conversation disappeared from the UI
and could throw errors when re-rendering. Now:

  - `messages` in the checkpoint is NEVER trimmed - the UI always shows
    everything, exactly like Phase 1.
  - This table tracks how much of that history has already been folded
    into a summary (`summarized_count`) and what that summary currently
    says.
  - call_model() uses this table to decide what to actually send to the
    LLM (summary + only the unsummarized tail), keeping the model's
    input bounded without ever losing anything from storage.
"""

from DB.connection import get_pool

def init_summary_context_table():
    """Idempotent - safe to call on every app startup."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_context (
                thread_id      TEXT PRIMARY KEY REFERENCES chat_sessions(thread_id) ON DELETE CASCADE,
                summary        TEXT NOT NULL DEFAULT '',
                summarized_count INTEGER NOT NULL DEFAULT 0,
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )


def get_summary_context(thread_id: str):
    """Returns (summary: str, summarized_count: int). Defaults to ("", 0)
    if this thread has never been summarized."""
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT summary, summarized_count FROM summary_context WHERE thread_id = %s",
            (thread_id,),
        ).fetchone()
    return (row[0], row[1]) if row else ("", 0)


def upsert_summary_context(thread_id: str, summary: str, summarized_count: int):
    """Insert or update the summary context for a given thread."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO summary_context (thread_id, summary, summarized_count, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (thread_id)
            DO UPDATE SET summary = EXCLUDED.summary,
                          summarized_count = EXCLUDED.summarized_count,
                          updated_at = now();
            """,
            (thread_id, summary, summarized_count),
        )
