"""
init_db.py

Run this ONCE (or any time you want, it's idempotent) before using the
chatbot or the monitor dashboard:

    python init_db.py

It creates:
  - LangGraph's checkpointer tables (checkpoints, checkpoint_blobs,
    checkpoint_writes, checkpoint_migrations) via PostgresSaver.setup()
  - LangGraph's long-term store tables (store, store_migrations) via
    PostgresStore.setup() - this is Phase 3's actual storage backend,
    now in active use (not just scaffolding)
  - This project's chat_sessions table (DB/chat_sessions.py)
  - This project's summary_context table (DB/summary_context.py) -
    Phase 2's running-summary storage, separate from checkpointed history
  - This project's profiles table (DB/profiles.py) - NEW in Phase 3,
    scopes long-term memory across chats. Also guarantees a "default"
    profile exists so the app never starts with zero profiles to select.
"""
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from DB.connection import get_conn_string, get_pool
from DB.users import init_users_table
from DB.chat_sessions import init_chat_sessions_table
from DB.summary_context import init_summary_context_table
from DB.profiles import init_profiles_table

DB_URI = get_conn_string()

with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    print("✅ checkpointer tables ready (checkpoints, checkpoint_blobs, checkpoint_writes)")

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
    print("✅ store tables ready (store, store_migrations) - now used by Phase 3 long-term memory")

# Initialize users table first
init_users_table()
print("✅ users table ready")

# Run migrations to ensure columns exist in existing db
pool = get_pool()
with pool.connection() as conn:
    conn.execute("ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS username TEXT NOT NULL DEFAULT 'default_user';")
    conn.execute("ALTER TABLE profiles ADD COLUMN IF NOT EXISTS username TEXT NOT NULL DEFAULT 'default_user';")
print("✅ migrations checked (username columns verified)")

init_chat_sessions_table()
print("✅ chat_sessions table ready")

init_summary_context_table()
print("✅ summary_context table ready")

init_profiles_table()
print("✅ profiles table ready (default profile ensured)")

print("\nDone. You can now run:")
print("  streamlit run Chatbot/app.py")
print("  streamlit run DB/monitor_app.py")