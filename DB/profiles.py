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
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                username    TEXT NOT NULL DEFAULT 'default_user'
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
            INSERT INTO profiles (profile_id, name, username)
            VALUES (%s, 'Default', 'default_user')
            ON CONFLICT (profile_id) DO NOTHING;
            """,
            (DEFAULT_PROFILE_ID,),
        )

def create_profile(username: str, name: str = "New Profile") -> str:
    profile_id = str(uuid.uuid4())
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            "INSERT INTO profiles (profile_id, name, username) VALUES (%s, %s, %s)",
            (profile_id, name, username),
        )
    return profile_id

def list_profiles(username: str):
    """Returns rows: (profile_id, name, created_at, updated_at)"""
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT profile_id, name, created_at, updated_at
            FROM profiles
            WHERE username = %s
            ORDER BY updated_at DESC
            """,
            (username,),
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

def delete_profile(profile_id: str, username: str):
    """
    Deletes the profile row and all its long-term memory notes from the
    `store` table. Refuses to delete the last remaining profile of the user.
    """
    pool = get_pool()
    with pool.connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM profiles WHERE username = %s", (username,)
        ).fetchone()[0]
        if count <= 1:
            raise ValueError("You must keep at least one profile.")

        conn.execute(
            "DELETE FROM store WHERE prefix[1] = %s", (profile_id,)
        )
        conn.execute("DELETE FROM profiles WHERE profile_id = %s AND username = %s", (profile_id, username))

def ensure_user_has_profile(username: str) -> str:
    """Ensures the user has at least one profile. Returns the active profile ID."""
    username = username.strip().lower()
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT profile_id FROM profiles WHERE username = %s LIMIT 1",
            (username,)
        ).fetchone()
        if row:
            return row[0]
        
        # If no profile, create a default one for this user
        profile_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO profiles (profile_id, name, username) VALUES (%s, 'Default', %s)",
            (profile_id, username),
        )
        return profile_id

