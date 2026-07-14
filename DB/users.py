"""
DB/users.py

Provides simple authentication, password hashing using SHA-256 with a salt,
and user record storage in the Postgres database.
"""
import hashlib
from DB.connection import get_pool

def init_users_table():
    """Idempotent - safe to call on every app startup."""
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username      TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
            """
        )


def _hash_password(password: str, username: str) -> str:
    """Combines password and username (acting as salt) and returns SHA-256 hex digest."""
    salted = f"{password}:{username.strip().lower()}"
    return hashlib.sha256(salted.encode("utf-8")).hexdigest()


def create_user(username: str, password: str) -> bool:
    """
    Creates a new user account.
    Returns True if successful, or False if the username is taken or empty.
    """
    username = username.strip().lower()
    password = password.strip()
    if not username or not password:
        return False
    
    pwd_hash = _hash_password(password, username)
    pool = get_pool()
    try:
        with pool.connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
                (username, pwd_hash),
            )
        return True
    except Exception:
        # Username exists or insertion failed
        return False


def verify_user(username: str, password: str) -> bool:
    """
    Verifies that the username exists and the password matches.
    """
    username = username.strip().lower()
    password = password.strip()
    if not username or not password:
        return False
    
    pwd_hash = _hash_password(password, username)
    pool = get_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE username = %s",
            (username,),
        ).fetchone()
    if row and row[0] == pwd_hash:
        return True
    return False
