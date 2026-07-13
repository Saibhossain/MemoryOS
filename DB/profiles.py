"""
DB/profiles.py

Long-term memory (Phase 3) is scoped to a "profile", not a thread — the
whole point is that it survives across different chats. This module owns
a small `profiles` table, structurally similar to `chat_sessions`, that
lets the Streamlit sidebar list, create, rename, and switch between
profiles. A "default" profile is auto-created on first run so the app
works immediately without forcing profile setup on anyone.
"""

import uuid
from DB.connection import get_pool

DEFAULT_PROFILE_ID = "default"

def init_profiles_table():
    """Idempotent - safe to call on every app startup."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id  TEXT PRIMARY KEY,
                name        TEXT NOT NULL DEFAULT 'Default',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_profiles_updated_at "
            "ON profiles (updated_at DESC);"
        )

    _ensure_default_profile()

def _ensure_default_profile():
    """Guarantees a 'default' profile always exists, so the app never
    starts in a state with zero profiles to select from."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            INSERT INTO profiles (profile_id, name)
            VALUES (%s, 'Default')
            ON CONFLICT (profile_id) DO NOTHING;
            """,
            (DEFAULT_PROFILE_ID,),
        )

def create_profile(name: str = "New Profile") -> str:
    profile_id = str(uuid.uuid4())
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO profiles (profile_id, name) VALUES (%s, %s)",
            (profile_id, name),
        )
    return profile_id

def list_profiles():
    """Returns rows: (profile_id, name, created_at, updated_at)"""
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT profile_id, name, created_at, updated_at
            FROM profiles
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return rows

def get_profile_name(profile_id: str) -> str:
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT name FROM profiles WHERE profile_id = %s", (profile_id,)
        ).fetchone()
    return row[0] if row else "Unknown Profile"

def rename_profile(profile_id: str, new_name: str):
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "UPDATE profiles SET name = %s, updated_at = now() WHERE profile_id = %s",
            (new_name, profile_id),
        )

def touch_profile(profile_id: str):
    """Bumps updated_at whenever a note is added/removed under this
    profile, so the sidebar can sort by recency like it does for chats."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "UPDATE profiles SET updated_at = now() WHERE profile_id = %s",
            (profile_id,),
        )

def delete_profile(profile_id: str):
    """
    Deletes the profile row and all its long-term memory notes from the
    `store` table. Refuses to delete the default profile - the app always
    needs at least one profile to fall back to.
    """
    if profile_id == DEFAULT_PROFILE_ID:
        raise ValueError("The default profile cannot be deleted.")

    pool = get_pool()
    with pool.connection() as conn:
        # store table's namespace column stores (profile_id, "notes") as
        # a Postgres array/path - filtering on the first element.
        conn.execute(
            "DELETE FROM store WHERE prefix[1] = %s", (profile_id,)
        )
        conn.execute("DELETE FROM profiles WHERE profile_id = %s", (profile_id,))

