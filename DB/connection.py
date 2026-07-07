import os

from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("\nDATABASE_URL is not set in the environment variables."
                    "DATABASE_URL not set. Create a .env file in the project root "
                    "(copy .env.example) with your Postgres connection string.\n"
)

_pool:ConnectionPool | None = None

def get_pool() -> ConnectionPool:
    """Return a process-wide connection pool, created once and reused.
 
    Streamlit re-runs your script top-to-bottom on every interaction, so
    without this global + reuse pattern you'd open a brand new pool of
    connections every time someone clicks a button.
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=DATABASE_URL,
            min_size=1,
            max_size=10,
            kwargs={"autocommit": True},
        )
    return _pool


def get_conn_string() -> str:
    """LangGraph's PostgresSaver / PostgresStore want a raw connection
    string (they manage their own connection internally), not a pool."""
    
    return DATABASE_URL