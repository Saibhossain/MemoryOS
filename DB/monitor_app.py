"""
DB/monitor_app.py

A modern, multi-page Streamlit dashboard for MemoryOS's Postgres backend.
Supports multi-user scoping, comprehensive visual analytics, and administrative controls.
"""
import sys
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DB.connection import get_pool
from DB.summary_context import get_summary_context
from DB.long_term_memory import list_notes 
from Chatbot.graph import build_graph
from Chatbot.utils import message_text, message_has_image_note

st.set_page_config(page_title="MemoryOS — DB Monitor", layout="wide", page_icon="🐘")
pool = get_pool()

@st.cache_resource
def get_graph():
    return build_graph()

graph = get_graph()

# -----------------------------------------------------------------------
# Styling & Theme Customizations
# -----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; }
        div[data-testid="stMetric"] {
            background: rgba(120, 120, 180, 0.04);
            border: 1px solid rgba(120, 120, 180, 0.12);
            border-radius: 8px;
            padding: 14px 18px;
        }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem; opacity: 0.85; }
        h1, h2, h3 { font-weight: 600; }
        .memoryos-pill {
            display: inline-block; padding: 3px 12px; border-radius: 12px;
            font-size: 0.75rem; font-weight: 600; margin-right: 6px;
        }
        .pill-green { background: rgba(31, 157, 85, 0.1); color: #1f9d55; }
        .pill-amber { background: rgba(217, 122, 25, 0.1); color: #d97a19; }
        .pill-blue  { background: rgba(37, 99, 235, 0.1); color: #2563eb; }
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
    with pool.connection() as conn:
        conn.execute(sql, params or ())

# -----------------------------------------------------------------------
# Sidebar Controls & Scope Filtering
# -----------------------------------------------------------------------
with st.sidebar:
    st.title("🐘 MemoryOS Dashboard")
    st.caption("Postgres Memory Monitor & Admin Tool")
    
    # Load all users dynamically to allow filtering scope
    try:
        users_df = run_query("SELECT username FROM users ORDER BY username;")
        if not users_df.empty:
            users_list = ["All Users"] + users_df["username"].tolist()
        else:
            users_list = ["All Users", "default_user"]
    except Exception:
        users_list = ["All Users", "default_user"]
        
    selected_user = st.selectbox("🎯 User Context Scope", options=users_list, index=0)
    
    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "🧵 Threads",
            "💾 Storage & Health",
            "🔌 Live Connections",
            "🛠️ SQL Console",
            "🧠 Long-term Memory",
        ],
    )
    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Build filtering SQL clauses
user_filter_clause = ""
user_filter_params = []
if selected_user != "All Users":
    user_filter_clause = " WHERE username = %s "
    user_filter_params = [selected_user]

# =========================================================================
# PAGE: Overview
# =========================================================================
if page == "📊 Overview":
    st.title("📊 Database Overview")
    st.markdown(f"Currently monitoring context scope: **{selected_user}**")
    
    # Core DB Stats
    db_size_df = run_query("SELECT pg_size_pretty(pg_database_size(current_database())) AS size;")
    version_df = run_query("SHOW server_version;")
    conn_count_df = run_query(
        "SELECT count(*) AS n FROM pg_stat_activity WHERE datname = current_database();"
    )
    
    # User Scoped Stats
    chats_sql = "SELECT count(*) AS n FROM chat_sessions" + (user_filter_clause if selected_user != "All Users" else "")
    chats_count_df = run_query(chats_sql, user_filter_params)
    
    cp_sql = """
        SELECT count(*) AS n FROM checkpoints c
        INNER JOIN chat_sessions cs ON c.thread_id = cs.thread_id
    """ + (user_filter_clause.replace("username", "cs.username") if selected_user != "All Users" else "")
    checkpoints_count_df = run_query(cp_sql, user_filter_params)
    
    profiles_sql = "SELECT count(*) AS n FROM profiles" + (user_filter_clause if selected_user != "All Users" else "")
    profiles_count_df = run_query(profiles_sql, user_filter_params)
    
    users_count_df = run_query("SELECT count(*) AS n FROM users;")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Database Size", db_size_df["size"].iloc[0])
    c2.metric("Postgres Version", version_df.iloc[0, 0].split(" ")[0])
    c3.metric("Active Connections", int(conn_count_df["n"].iloc[0]))
    c4.metric("Total System Users", int(users_count_df["n"].iloc[0]))
    
    d1, d2, d3 = st.columns(3)
    d1.metric(f"Chats ({selected_user})", int(chats_count_df["n"].iloc[0]))
    d2.metric(f"Checkpoints ({selected_user})", int(checkpoints_count_df["n"].iloc[0]))
    d3.metric(f"Profiles ({selected_user})", int(profiles_count_df["n"].iloc[0]))

    st.divider()

    # Storage Statistics
    st.subheader("📦 Database Storage Analytics")
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

    if not table_sizes.empty:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig = px.bar(
                table_sizes, x="table_name", y="total_bytes", text="total_size",
                title="Disk Usage per Table", color="table_name",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Bytes")
            st.plotly_chart(fig, use_container_width=True)
        with col_b:
            fig_pie = px.pie(
                table_sizes, names="table_name", values="total_bytes",
                title="Storage Space Share", hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("📈 Row Counts & Table Activity")
        fig_rows = px.bar(
            table_sizes.sort_values("approx_row_count", ascending=True),
            x="approx_row_count", y="table_name", orientation="h",
            title="Approximate Row Counts by Table",
            color="table_name", color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_rows.update_layout(yaxis_title="", xaxis_title="Rows", showlegend=False)
        st.plotly_chart(fig_rows, use_container_width=True)
    else:
        st.info("No tables detected. Run the chatbot first to initialize the database.")

    # Checkpoints Count per Thread
    st.subheader("🧵 Checkpoint Distribution (Top Threads)")
    cp_list_sql = """
        SELECT c.thread_id, COALESCE(cs.title, c.thread_id) AS title, cs.username, count(*) AS checkpoint_count
        FROM checkpoints c
        LEFT JOIN chat_sessions cs ON cs.thread_id = c.thread_id
    """
    if selected_user != "All Users":
        cp_list_sql += " WHERE cs.username = %s "
    cp_list_sql += """
        GROUP BY c.thread_id, cs.title, cs.username
        ORDER BY checkpoint_count DESC
        LIMIT 15;
    """
    cp_df = run_query(cp_list_sql, user_filter_params)
    if not cp_df.empty:
        fig2 = px.bar(
            cp_df, x="title", y="checkpoint_count", color="username",
            title="Top 15 Threads by Saved States (Checkpoints)"
        )
        fig2.update_layout(xaxis_title="", yaxis_title="Checkpoints")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No thread checkpoints found for this user.")

# =========================================================================
# PAGE: Threads (list + per-thread deep dive)
# =========================================================================
elif page == "🧵 Threads":
    st.title("🧵 Thread Inspector")
    st.markdown(f"Viewing active threads for: **{selected_user}**")

    chats_sql = """
        SELECT thread_id, title, username, created_at, updated_at, message_count
        FROM chat_sessions
    """
    if selected_user != "All Users":
        chats_sql += " WHERE username = %s "
    chats_sql += " ORDER BY updated_at DESC;"
    
    chats_df = run_query(chats_sql, user_filter_params)

    if chats_df.empty:
        st.info(f"No chats found for user scope '{selected_user}'. Send a chat message first.")
        st.stop()

    search = st.text_input("🔍 Filter thread titles", "")
    filtered = chats_df[chats_df["title"].str.contains(search, case=False, na=False)] if search else chats_df

    st.dataframe(
        filtered[["username", "title", "thread_id", "created_at", "updated_at", "message_count"]],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.header("🔎 Inspect Specific Conversation Thread")

    options = filtered["title"] + "  ·  (" + filtered["username"] + ")  ·  " + filtered["thread_id"].str.slice(0, 8)
    if options.empty:
        st.info("No threads match your search filter.")
        st.stop()

    choice = st.selectbox("Select Thread", options.tolist())
    chosen_idx = options.tolist().index(choice)
    thread_id = filtered.iloc[chosen_idx]["thread_id"]
    title = filtered.iloc[chosen_idx]["title"]

    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    messages = state.values.get("messages", [])
    summary, summarized_count = get_summary_context(thread_id)

    st.subheader(f"📄 Thread: {title}")
    tags = []
    if summary:
        tags.append('<span class="memoryos-pill pill-blue">🧵 Summarized Context</span>')
    if any(message_has_image_note(m) for m in messages):
        tags.append('<span class="memoryos-pill pill-amber">🖼️ Vision Logs</span>')
    tags.append('<span class="memoryos-pill pill-green">Active Session</span>')
    st.markdown(" ".join(tags), unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Messages Stored", len(messages))
    m2.metric("Active Window size", len(messages) - summarized_count)
    m3.metric("Summarized Count", summarized_count)

    cp_thread_df = run_query(
        "SELECT checkpoint_id, parent_checkpoint_id FROM checkpoints WHERE thread_id = %s ORDER BY checkpoint_id;",
        (thread_id,),
    )
    m4.metric("Checkpoint Chain Length", len(cp_thread_df))

    if summary:
        with st.expander("📋 Current Active Summary Context", expanded=False):
            st.write(summary)

    # Checkpoint scatter-plot map
    st.subheader("⛓️ Checkpoint Chain Map")
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
            title="Saved State Chain Snapshots",
            yaxis=dict(visible=False), xaxis_title="Timeline / Turn Sequence", height=200,
            showlegend=False, margin=dict(t=30, b=30, l=10, r=10)
        )
        st.plotly_chart(chain_fig, use_container_width=True)

    # Message composition analytics
    if messages:
        st.subheader("📊 Thread Message Composition Analytics")
        msg_rows = []
        cumulative = 0
        for i, m in enumerate(messages):
            text = message_text(m)
            cumulative += len(text)
            msg_rows.append({
                "index": i + 1,
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
            fig_role = px.pie(role_counts, names="role", values="count", title="User vs Assistant Distribution", hole=0.4)
            st.plotly_chart(fig_role, use_container_width=True)
        with cc2:
            fig_len = px.bar(msg_df, x="index", y="length", color="role", title="Message Characters per Turn")
            fig_len.update_layout(xaxis_title="Turn Sequence", yaxis_title="Character Length")
            st.plotly_chart(fig_len, use_container_width=True)

        fig_growth = px.area(
            msg_df, x="index", y="cumulative_chars",
            title="Total Conversation Scale Growth Over Time",
        )
        fig_growth.update_layout(xaxis_title="Turn Sequence", yaxis_title="Cumulative Characters")
        if summarized_count > 0:
            fig_growth.add_vline(
                x=summarized_count + 0.5, line_dash="dash", line_color="orange",
                annotation_text="Context Summarization Threshold",
            )
        st.plotly_chart(fig_growth, use_container_width=True)

        st.subheader("📜 Thread Message History Logs")
        for i, m in enumerate(messages):
            role_label = "🧑 User" if m.type == "human" else "🤖 Assistant"
            marker = " 🧵" if i < summarized_count else ""
            with st.expander(f"{role_label}{marker} — Turn {i + 1}"):
                st.text(message_text(m))

# =========================================================================
# PAGE: Storage & Health
# =========================================================================
elif page == "💾 Storage & Health":
    st.title("💾 Storage, Performance & Maintenance")

    st.subheader("🔍 Table & Index Size Allocation")
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

    st.subheader("♻️ Table Maintenance & Vacuum Metrics")
    vacuum_df = run_query(
        """
        SELECT relname AS table_name, n_live_tup AS live_rows, n_dead_tup AS dead_rows,
               last_vacuum, last_autovacuum, last_analyze, last_autoanalyze
        FROM pg_stat_user_tables
        ORDER BY dead_rows DESC;
        """
    )
    st.dataframe(vacuum_df, use_container_width=True, hide_index=True)

    if not vacuum_df.empty:
        fig_dead = px.bar(
            vacuum_df, x="table_name", y="dead_rows",
            title="Uncleaned / Dead Rows per Table (Vacuum Candidates)",
            color="table_name", color_discrete_sequence=px.colors.qualitative.Warm
        )
        st.plotly_chart(fig_dead, use_container_width=True)

    st.divider()
    st.subheader("🛠️ Database Admin Actions")
    st.warning("These operations run active queries against the production tables. Execute with care.")

    if not size_df.empty:
        maint_table = st.selectbox("Select Table to Optimize", size_df["table_name"].tolist())
        if st.button(f"⚡ Optimize (VACUUM ANALYZE) `{maint_table}`", use_container_width=True):
            try:
                run_action(f"VACUUM ANALYZE {maint_table};")
                st.success(f"Successfully optimized and updated stats on `{maint_table}`.")
                st.rerun()
            except Exception as e:
                st.error(f"Maintenance failed: {e}")

# =========================================================================
# PAGE: Live Connection Activity
# =========================================================================
elif page == "🔌 Live Connections":
    st.title("🔌 Live Postgres Connections")
    st.markdown("Monitor real-time application queries and active client threads.")

    activity_df = run_query(
        """
        SELECT pid, usename, application_name, client_addr, state, query, query_start
        FROM pg_stat_activity
        WHERE datname = current_database()
        ORDER BY query_start DESC NULLS LAST;
        """
    )
    st.dataframe(activity_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🛑 Terminate Client PID Session")
    st.caption("Forcibly disconnect hung connections or stale Streamlit client pools.")
    
    if not activity_df.empty:
        pid_choice = st.selectbox("Select PID Connection to Kill", activity_df["pid"].tolist())
        if st.button(f"💥 Terminate PID {pid_choice}", type="primary", use_container_width=True):
            try:
                run_action("SELECT pg_terminate_backend(%s);", (pid_choice,))
                st.success(f"Connection thread PID {pid_choice} successfully killed.")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to close connection: {e}")

# =========================================================================
# PAGE: SQL Console
# =========================================================================
elif page == "🛠️ SQL Console":
    st.title("🛠️ SQL Console Query Runner")

    admin_mode = st.checkbox("⚠️ Enable DB Schema Modifications (Allow INSERT/UPDATE/DELETE/DDL)")
    if admin_mode:
        st.warning("Write / DDL permissions enabled. Ensure queries are checked before executing.")

    query_text = st.text_area(
        "Enter SQL Statement",
        value="SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT 10;",
        height=180,
    )

    if st.button("▶ Run SQL Query", use_container_width=True):
        stripped = query_text.strip().lower()
        if not admin_mode and not any(stripped.startswith(prefix) for prefix in ["select", "show", "explain"]):
            st.error("Operation Denied: Query modifications are blocked. Enable write mode first.")
        else:
            try:
                if any(stripped.startswith(prefix) for prefix in ["select", "show", "explain"]):
                    result_df = run_query(query_text)
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                    st.success(f"Query returned {len(result_df)} rows.")
                else:
                    run_action(query_text)
                    st.success("DDL/DML query executed successfully.")
            except Exception as e:
                st.error(f"SQL execution error: {e}")

# =========================================================================
# PAGE: Long-term Memory
# =========================================================================
elif page == "🧠 Long-term Memory":
    st.title("🧠 Long-term Memory Profile Explorer")
    st.markdown(f"Viewing active memory scope for: **{selected_user}**")

    # Load profiles scoped to user context
    profiles_sql = "SELECT profile_id, name, created_at, updated_at FROM profiles"
    if selected_user != "All Users":
        profiles_sql += " WHERE username = %s "
    profiles_sql += " ORDER BY updated_at DESC;"
    profiles_all = run_query(profiles_sql, user_filter_params)

    if profiles_all.empty:
        st.info("No profiles detected for the selected user scope.")
        st.stop()

    profile_rows = []
    for idx, row in profiles_all.iterrows():
        pid = row["profile_id"]
        notes = list_notes(pid)
        profile_rows.append({
            "profile_id": pid,
            "name": row["name"],
            "note_count": len(notes),
            "updated_at": row["updated_at"],
        })
    profile_summary_df = pd.DataFrame(profile_rows)

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Scoped Profiles", len(profile_summary_df))
    m2.metric("Total Scoped Notes", int(profile_summary_df["note_count"].sum()))
    m3.metric("Average Notes / Profile", round(profile_summary_df["note_count"].mean(), 1) if len(profile_summary_df) else 0)

    if profile_summary_df["note_count"].sum() > 0:
        fig_notes = px.bar(
            profile_summary_df.sort_values("note_count", ascending=True),
            x="note_count", y="name", orientation="h",
            title="Saved Facts Count per Memory Profile",
            color="name", color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_notes.update_layout(xaxis_title="Notes Count", yaxis_title="", showlegend=False)
        st.plotly_chart(fig_notes, use_container_width=True)

    st.dataframe(
        profile_summary_df[["name", "profile_id", "note_count", "updated_at"]],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.header("🔎 Browse Scoped Notes")

    profile_choice_labels = {p[0]: p[1] for p in profiles_all.values}
    selected_profile_id = st.selectbox(
        "Select Profile Context",
        options=list(profile_choice_labels.keys()),
        format_func=lambda pid: profile_choice_labels[pid],
    )

    notes = list_notes(selected_profile_id)
    if not notes:
        st.info("No memory notes saved under the selected profile context.")
    else:
        notes_df = pd.DataFrame(notes)

        search = st.text_input("🔍 Search memory content", "")
        filtered_notes = notes_df[notes_df["text"].str.contains(search, case=False, na=False)] if search else notes_df

        st.dataframe(
            filtered_notes[["text", "created_at", "source_thread_id", "key"]],
            use_container_width=True, hide_index=True,
        )

        with st.expander("🛠️ Full Notes Diagnostic List (With IDs)", expanded=False):
            for n in notes:
                st.markdown(f"**[{n['created_at']}]** {n['text']}")
                st.caption(f"Note UUID: `{n['key']}` · Source Thread ID: `{n['source_thread_id']}`")