"""
DB/monitor_app.py

A read-only Streamlit dashboard for watching what LangGraph + your chatbot
are actually doing to Postgres: table sizes, row counts, per-thread state
growth, live connections, and a raw SQL browser.

Run with:
    streamlit run DB/monitor_app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.express as px
import streamlit as st

from DB.connection import get_pool

st.set_page_config(page_title="MemoryOS - DB Monitor", layout="wide")

pool = get_pool()


def run_query(sql: str, params=None) -> pd.DataFrame:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            cols = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


st.title("🐘 MemoryOS — Postgres Monitoring Dashboard")
st.caption(
    "Live view into how the agent's short-term (checkpointer) and "
    "app-level (chat_sessions) memory is stored."
)

if st.button("🔄 Refresh now"):
    st.rerun()

# ---------------------------------------------------------------------
# 1. Database-level overview
# ---------------------------------------------------------------------
st.header("Database overview")

col1, col2, col3 = st.columns(3)

db_size_df = run_query(
    "SELECT pg_size_pretty(pg_database_size(current_database())) AS size;"
)
version_df = run_query("SHOW server_version;")
conn_count_df = run_query(
    "SELECT count(*) AS n FROM pg_stat_activity WHERE datname = current_database();"
)

with col1:
    st.metric("Database size", db_size_df["size"].iloc[0])
with col2:
    st.metric("Postgres version", version_df.iloc[0, 0])
with col3:
    st.metric("Active connections", int(conn_count_df["n"].iloc[0]))

# ---------------------------------------------------------------------
# 2. Table sizes - where is memory actually going?
# ---------------------------------------------------------------------
st.header("Table sizes")

table_sizes = run_query(
    """
    SELECT
        relname AS table_name,
        pg_total_relation_size(relid) AS total_bytes,
        pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
        n_live_tup AS approx_row_count
    FROM pg_catalog.pg_stat_user_tables
    ORDER BY total_bytes DESC;
    """
)

if not table_sizes.empty:
    fig = px.bar(
        table_sizes,
        x="table_name",
        y="total_bytes",
        text="total_size",
        title="Disk usage per table (bytes)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        table_sizes[["table_name", "total_size", "approx_row_count"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No tables found yet. Run the chatbot at least once first.")

# ---------------------------------------------------------------------
# 3. Chat sessions overview (application-level memory index)
# ---------------------------------------------------------------------
st.header("Chat sessions")

try:
    chats_df = run_query(
        """
        SELECT thread_id, title, created_at, updated_at, message_count
        FROM chat_sessions
        ORDER BY updated_at DESC;
        """
    )
    st.dataframe(chats_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"chat_sessions table not available yet: {e}")

# ---------------------------------------------------------------------
# 4. Checkpoint growth per thread - the actual short-term memory chain
# ---------------------------------------------------------------------
st.header("Checkpoints per thread (short-term memory chain)")

try:
    cp_df = run_query(
        """
        SELECT
            thread_id,
            count(*) AS checkpoint_count
        FROM checkpoints
        GROUP BY thread_id
        ORDER BY checkpoint_count DESC;
        """
    )
    if not cp_df.empty:
        fig2 = px.bar(
            cp_df,
            x="thread_id",
            y="checkpoint_count",
            title="Number of saved checkpoints (state snapshots) per thread",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No checkpoints yet - send a message in the chatbot first.")
except Exception as e:
    st.warning(f"checkpoints table not available yet: {e}")

# ---------------------------------------------------------------------
# 5. Long-term memory store browser (Phase 3, empty until you build it)
# ---------------------------------------------------------------------
st.header("Long-term memory (store table)")

try:
    store_df = run_query(
        """
        SELECT prefix AS namespace, key, value, updated_at
        FROM store
        ORDER BY updated_at DESC
        LIMIT 100;
        """
    )
    if store_df.empty:
        st.info("Empty - this fills up once you implement Phase 3 (long-term memory).")
    else:
        st.dataframe(store_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"store table not available yet: {e}")

# ---------------------------------------------------------------------
# 6. Live activity
# ---------------------------------------------------------------------
st.header("Live connections (pg_stat_activity)")

activity_df = run_query(
    """
    SELECT pid, usename, application_name, state, query, query_start
    FROM pg_stat_activity
    WHERE datname = current_database()
    ORDER BY query_start DESC NULLS LAST;
    """
)
st.dataframe(activity_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# 7. Raw SQL browser (read-only)
# ---------------------------------------------------------------------
st.header("Run a read-only SQL query")
st.caption("Only SELECT statements are allowed here, as a safety guard.")

query_text = st.text_area(
    "SQL",
    value="SELECT * FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 20;",
    height=100,
)

if st.button("Run query"):
    stripped = query_text.strip().lower()
    if not stripped.startswith("select"):
        st.error("Only SELECT statements are allowed in this dashboard.")
    else:
        try:
            result_df = run_query(query_text)
            st.dataframe(result_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query failed: {e}")