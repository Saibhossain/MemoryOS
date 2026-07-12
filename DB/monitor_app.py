"""
DB/monitor_app.py  (Phase 2, full redesign)

A modern, multi-page Streamlit dashboard for MemoryOS's Postgres backend.

Pages (sidebar navigation):
  - Overview        : DB-wide health metrics, table size charts
  - Threads          : searchable list of every chat + per-thread deep dive
  - Storage & Health : table/index sizes, vacuum stats, maintenance actions
  - Live Activity    : pg_stat_activity, with a "terminate connection" control
  - SQL Console       : guarded query runner (read-only by default, opt-in write mode)
  - Long-term Memory : Phase 3 store table browser (empty until you build it)

Run with:
    streamlit run DB/monitor_app.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from DB.connection import get_pool
from DB.summary_context import get_summary_context
from Chatbot.graph import build_graph
from Chatbot.utils import message_text, message_has_image_note

st.set_page_config(page_title="MemoryOS — DB Monitor", layout="wide", page_icon="🐘")

pool = get_pool()


@st.cache_resource
def get_graph():
    return build_graph()


graph = get_graph()

# -----------------------------------------------------------------------
# Styling — a bit of custom CSS to move away from default Streamlit look
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        div[data-testid="stMetric"] {
            background: rgba(120, 120, 180, 0.08);
            border: 1px solid rgba(120, 120, 180, 0.18);
            border-radius: 12px;
            padding: 12px 16px;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.8; }
        h1, h2, h3 { font-weight: 650; }
        .memoryos-pill {
            display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; margin-right: 6px;
        }
        .pill-green { background: #1f9d551a; color: #1f9d55; }
        .pill-amber { background: #d97a191a; color: #d97a19; }
        .pill-blue  { background: #2563eb1a; color: #2563eb; }
    </style>
    """,
    unsafe_allow_html=True,
)


def run_query(sql: str, params=None) -> pd.DataFrame:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            cols = [d.name for d in cur.description] if cur.description else []
            rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def run_action(sql: str, params=None):
    """For statements with no result set (VACUUM, pg_terminate_backend, etc).
    Pool connections already run with autocommit=True, which VACUUM requires."""
    with pool.connection() as conn:
        conn.execute(sql, params or ())


# -----------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("🐘 MemoryOS")
    st.caption("Postgres memory monitor")
    page = st.radio(
        "Navigate",
        [
            "📊 Overview",
            "🧵 Threads",
            "💾 Storage & Health",
            "🔌 Live Activity",
            "🛠️ SQL Console",
            "🧠 Long-term Memory",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    if st.button("🔄 Refresh all data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# =========================================================================
# PAGE: Overview
# =========================================================================
if page == "📊 Overview":
    st.title("📊 Database Overview")

    db_size_df = run_query("SELECT pg_size_pretty(pg_database_size(current_database())) AS size;")
    version_df = run_query("SHOW server_version;")
    conn_count_df = run_query(
        "SELECT count(*) AS n FROM pg_stat_activity WHERE datname = current_database();"
    )
    chats_count_df = run_query("SELECT count(*) AS n FROM chat_sessions;")
    checkpoints_count_df = run_query("SELECT count(*) AS n FROM checkpoints;")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Database size", db_size_df["size"].iloc[0])
    c2.metric("Postgres version", version_df.iloc[0, 0].split(" ")[0])
    c3.metric("Active connections", int(conn_count_df["n"].iloc[0]))
    c4.metric("Total chats", int(chats_count_df["n"].iloc[0]))
    c5.metric("Total checkpoints", int(checkpoints_count_df["n"].iloc[0]))

    st.divider()

    table_sizes = run_query(
        """
        SELECT relname AS table_name,
               pg_total_relation_size(relid) AS total_bytes,
               pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
               n_live_tup AS approx_row_count
        FROM pg_catalog.pg_stat_user_tables
        ORDER BY total_bytes DESC;
        """
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        if not table_sizes.empty:
            fig = px.bar(
                table_sizes, x="table_name", y="total_bytes", text="total_size",
                title="Disk usage per table", color="table_name",
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Bytes")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tables found yet. Run the chatbot at least once first.")
    with col_b:
        if not table_sizes.empty:
            fig_pie = px.pie(
                table_sizes, names="table_name", values="total_bytes",
                title="Storage share by table", hole=0.45,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Row counts by table")
    if not table_sizes.empty:
        fig_rows = px.bar(
            table_sizes.sort_values("approx_row_count", ascending=True),
            x="approx_row_count", y="table_name", orientation="h",
            title="Approximate row counts",
        )
        fig_rows.update_layout(yaxis_title="", xaxis_title="Rows")
        st.plotly_chart(fig_rows, use_container_width=True)

    st.subheader("Checkpoints per thread")
    cp_df = run_query(
        """
        SELECT c.thread_id, COALESCE(cs.title, c.thread_id) AS title, count(*) AS checkpoint_count
        FROM checkpoints c
        LEFT JOIN chat_sessions cs ON cs.thread_id = c.thread_id
        GROUP BY c.thread_id, cs.title
        ORDER BY checkpoint_count DESC
        LIMIT 20;
        """
    )
    if not cp_df.empty:
        fig2 = px.bar(cp_df, x="title", y="checkpoint_count", title="Top 20 threads by checkpoint count")
        fig2.update_layout(xaxis_title="", yaxis_title="Checkpoints")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No checkpoints yet — send a message in the chatbot first.")


# =========================================================================
# PAGE: Threads (list + per-thread deep dive)
# =========================================================================
elif page == "🧵 Threads":
    st.title("🧵 Threads")

    chats_df = run_query(
        """
        SELECT thread_id, title, created_at, updated_at, message_count
        FROM chat_sessions
        ORDER BY updated_at DESC;
        """
    )

    if chats_df.empty:
        st.info("No chats yet. Go send a message in the chatbot first.")
        st.stop()

    search = st.text_input("🔍 Filter by title", "")
    filtered = chats_df[chats_df["title"].str.contains(search, case=False, na=False)] if search else chats_df

    st.dataframe(
        filtered[["title", "thread_id", "created_at", "updated_at", "message_count"]],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.header("🔎 Inspect a thread")

    options = filtered["title"] + "  ·  " + filtered["thread_id"].str.slice(0, 8)
    if options.empty:
        st.info("No threads match your filter.")
        st.stop()

    choice = st.selectbox("Choose a thread", options.tolist())
    chosen_idx = options.tolist().index(choice)
    thread_id = filtered.iloc[chosen_idx]["thread_id"]
    title = filtered.iloc[chosen_idx]["title"]

    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    messages = state.values.get("messages", [])
    summary, summarized_count = get_summary_context(thread_id)

    st.subheader(f"📄 {title}")
    tags = []
    if summary:
        tags.append('<span class="memoryos-pill pill-blue">🧵 summarized</span>')
    if any(message_has_image_note(m) for m in messages):
        tags.append('<span class="memoryos-pill pill-amber">🖼️ has image analysis</span>')
    tags.append('<span class="memoryos-pill pill-green">live</span>')
    st.markdown(" ".join(tags), unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total messages (stored)", len(messages))
    m2.metric("Messages sent to model", len(messages) - summarized_count)
    m3.metric("Summarized count", summarized_count)

    cp_thread_df = run_query(
        "SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id;",
        (thread_id,),
    )
    m4.metric("Checkpoints (chain length)", len(cp_thread_df))

    try:
        blob_df = run_query(
            "SELECT pg_size_pretty(sum(pg_column_size(blob))) AS size, count(*) AS n "
            "FROM checkpoint_blobs WHERE thread_id = %s;",
            (thread_id,),
        )
        if not blob_df.empty and blob_df["size"].iloc[0]:
            st.caption(f"State blob storage for this thread: **{blob_df['size'].iloc[0]}** across {blob_df['n'].iloc[0]} blob rows.")
    except Exception:
        pass

    if summary:
        with st.expander("📋 Current running summary", expanded=False):
            st.write(summary)

    # ---- Checkpoint chain visualization ----
    st.subheader("Checkpoint chain")
    if not cp_thread_df.empty:
        chain_fig = go.Figure()
        x_vals = list(range(len(cp_thread_df)))
        chain_fig.add_trace(go.Scatter(
            x=x_vals, y=[1] * len(x_vals), mode="markers+lines",
            marker=dict(size=14, color="#2563eb"),
            line=dict(color="#93c5fd", width=2),
            text=[cid[:8] for cid in cp_thread_df["checkpoint_id"]],
            hovertemplate="Checkpoint %{text}<extra></extra>",
        ))
        chain_fig.update_layout(
            title=f"{len(cp_thread_df)} checkpoints — each one a saved state snapshot, linked to its parent",
            yaxis=dict(visible=False), xaxis_title="Sequence", height=220,
            showlegend=False,
        )
        st.plotly_chart(chain_fig, use_container_width=True)
    else:
        st.info("No checkpoints recorded for this thread yet.")

    # ---- Message composition charts ----
    if messages:
        st.subheader("Message composition")
        msg_rows = []
        cumulative = 0
        for i, m in enumerate(messages):
            text = message_text(m)
            cumulative += len(text)
            msg_rows.append({
                "index": i,
                "role": "user" if m.type == "human" else "assistant",
                "length": len(text),
                "cumulative_chars": cumulative,
                "has_image_note": message_has_image_note(m),
            })
        msg_df = pd.DataFrame(msg_rows)

        cc1, cc2 = st.columns(2)
        with cc1:
            role_counts = msg_df["role"].value_counts().reset_index()
            role_counts.columns = ["role", "count"]
            fig_role = px.pie(role_counts, names="role", values="count", title="User vs assistant messages", hole=0.45)
            st.plotly_chart(fig_role, use_container_width=True)
        with cc2:
            fig_len = px.bar(msg_df, x="index", y="length", color="role", title="Message length by turn (chars)")
            fig_len.update_layout(xaxis_title="Turn #", yaxis_title="Characters")
            st.plotly_chart(fig_len, use_container_width=True)

        fig_growth = px.area(
            msg_df, x="index", y="cumulative_chars",
            title="Cumulative conversation size over time",
        )
        fig_growth.update_layout(xaxis_title="Turn #", yaxis_title="Cumulative characters")
        if summarized_count > 0:
            fig_growth.add_vline(
                x=summarized_count - 0.5, line_dash="dash", line_color="orange",
                annotation_text="summarization boundary",
            )
        st.plotly_chart(fig_growth, use_container_width=True)

        st.subheader("Full message log")
        for i, m in enumerate(messages):
            role_label = "🧑 User" if m.type == "human" else "🤖 Assistant"
            marker = " 🧵" if i < summarized_count else ""
            with st.expander(f"{role_label}{marker} — turn {i + 1}"):
                st.write(message_text(m))


# =========================================================================
# PAGE: Storage & Health
# =========================================================================
elif page == "💾 Storage & Health":
    st.title("💾 Storage & Health")

    st.subheader("Table & index sizes")
    size_df = run_query(
        """
        SELECT
            relname AS table_name,
            pg_size_pretty(pg_relation_size(relid)) AS table_size,
            pg_size_pretty(pg_total_relation_size(relid) - pg_relation_size(relid)) AS index_size,
            pg_size_pretty(pg_total_relation_size(relid)) AS total_size
        FROM pg_catalog.pg_stat_user_tables
        ORDER BY pg_total_relation_size(relid) DESC;
        """
    )
    st.dataframe(size_df, use_container_width=True, hide_index=True)

    st.subheader("Vacuum / bloat health")
    vacuum_df = run_query(
        """
        SELECT relname AS table_name, n_live_tup, n_dead_tup,
               last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY n_dead_tup DESC;
        """
    )
    st.dataframe(vacuum_df, use_container_width=True, hide_index=True)

    if not vacuum_df.empty:
        fig_dead = px.bar(
            vacuum_df, x="table_name", y="n_dead_tup",
            title="Dead tuples per table (candidates for vacuuming)",
        )
        st.plotly_chart(fig_dead, use_container_width=True)

    st.divider()
    st.subheader("🛠️ Maintenance actions")
    st.caption("These run real commands against your database. Use deliberately.")

    maint_table = st.selectbox(
        "Table to VACUUM ANALYZE",
        size_df["table_name"].tolist() if not size_df.empty else [],
    )
    if st.button(f"Run VACUUM ANALYZE on `{maint_table}`"):
        try:
            run_action(f"VACUUM ANALYZE {maint_table};")
            st.success(f"VACUUM ANALYZE completed on {maint_table}.")
        except Exception as e:
            st.error(f"Failed: {e}")


# =========================================================================
# PAGE: Live Activity
# =========================================================================
elif page == "🔌 Live Activity":
    st.title("🔌 Live Connections")

    activity_df = run_query(
        """
        SELECT pid, usename, application_name, state, query, query_start
        FROM pg_stat_activity
        WHERE datname = current_database()
        ORDER BY query_start DESC NULLS LAST;
        """
    )
    st.dataframe(activity_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🛑 Terminate a connection")
    st.caption(
        "Useful if a Streamlit rerun leaked a connection or a query is stuck. "
        "Terminating your own active session's pid will disconnect this dashboard."
    )
    if not activity_df.empty:
        pid_choice = st.selectbox("Choose a PID to terminate", activity_df["pid"].tolist())
        if st.button(f"Terminate PID {pid_choice}", type="primary"):
            try:
                run_action("SELECT pg_terminate_backend(%s);", (pid_choice,))
                st.success(f"Terminated PID {pid_choice}.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")


# =========================================================================
# PAGE: SQL Console
# =========================================================================
elif page == "🛠️ SQL Console":
    st.title("🛠️ SQL Console")

    admin_mode = st.checkbox(
        "⚠️ Enable write mode (allow INSERT/UPDATE/DELETE/DDL, not just SELECT)"
    )
    if admin_mode:
        st.warning(
            "Write mode is on. Queries here run directly against your database "
            "with no undo. Double-check before running anything destructive."
        )

    query_text = st.text_area(
        "SQL",
        value="SELECT * FROM checkpoints ORDER BY checkpoint_id DESC LIMIT 20;",
        height=140,
    )

    if st.button("▶ Run query"):
        stripped = query_text.strip().lower()
        if not admin_mode and not stripped.startswith("select"):
            st.error("Only SELECT statements are allowed unless write mode is enabled.")
        else:
            try:
                if stripped.startswith("select"):
                    result_df = run_query(query_text)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.caption(f"{len(result_df)} rows returned.")
                else:
                    run_action(query_text)
                    st.success("Statement executed successfully.")
            except Exception as e:
                st.error(f"Query failed: {e}")


# =========================================================================
# PAGE: Long-term Memory (Phase 3 placeholder)
# =========================================================================
elif page == "🧠 Long-term Memory":
    st.title("🧠 Long-term Memory (Store)")
    st.caption(
        "This table is created by LangGraph's PostgresStore but stays empty "
        "until Phase 3 (cross-thread, durable memory) is implemented."
    )

    try:
        store_df = run_query(
            """
            SELECT prefix AS namespace, key, value, updated_at
            FROM store
            ORDER BY updated_at DESC
            LIMIT 200;
            """
        )
        if store_df.empty:
            st.info("Empty — this fills up once Phase 3 is built.")
        else:
            namespaces = store_df["namespace"].unique().tolist()
            ns_filter = st.multiselect("Filter by namespace", namespaces, default=namespaces)
            st.dataframe(
                store_df[store_df["namespace"].isin(ns_filter)],
                use_container_width=True, hide_index=True,
            )
    except Exception as e:
        st.warning(f"store table not available yet: {e}")