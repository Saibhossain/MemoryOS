# Phase 1 — Short-Term Memory (Explained)

**Goal of Phase 1:** build a fully working local chatbot (Ollama + Qwen) whose
conversation memory lives in Postgres instead of RAM, using LangGraph's
`PostgresSaver` checkpointer. No RAG, no embeddings — that's deliberately
out of scope until Phase 2/3, so this phase is 100% about understanding
*state persistence*, not retrieval.

➡️ Next: [phase_2_explain.md](./phase_2_explain.md)

---

## 1. The two kinds of memory we're dealing with

| | Checkpointer (`PostgresSaver`) | `chat_sessions` table |
|---|---|---|
| What it stores | The full message list, every turn | Chat title, timestamps, message count |
| Who writes it | LangGraph, automatically, on every `.invoke()` | Our own code, explicitly, after each turn |
| Keyed by | `thread_id` | `thread_id` (same key, separate table) |
| Purpose | Feeds the model its own conversation history | Feeds the sidebar's chat list |

LangGraph's checkpointer was built to answer *"what's the state of thread X"* —
it was never meant to answer *"show me all my chats."* That's why the project
has a second, much simpler table (`chat_sessions`) sitting next to it. This
separation is the single most important idea in Phase 1.

---

## 2. File-by-file walkthrough

```bash
    MemoryOS/
    ├── .env
    ├── init_db.py
    ├── DB/
    │   ├── __init__.py
    │   ├── connection.py
    │   ├── chat_sessions.py
    │   └── monitor_app.py
    └── Chatbot/
        ├── __init__.py
        ├── graph.py
        ├── utils.py
        └── app.py
```

### `DB/connection.py` — one shared connection pool

```python
_pool: ConnectionPool | None = None

def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo=DATABASE_URL, min_size=1, max_size=10, kwargs={"autocommit": True})
    return _pool
```

**Why this matters:** Streamlit re-runs your entire script top-to-bottom on
every click, text input, or rerun. Without this "create once, cache globally"
pattern, every single interaction would open a brand-new pool of Postgres
connections — you'd leak connections fast and eventually hit
`max_connections`. `get_pool()` guarantees the whole app shares one pool for
its entire process lifetime.

`get_conn_string()` exists separately because LangGraph's `PostgresSaver`
and `PostgresStore` don't take a pool — they manage their own single
connection internally, so they just want the raw URL string.

---

### `DB/chat_sessions.py` — the application-level "index" of conversations

This is a plain CRUD module against one custom table:

```sql
CREATE TABLE chat_sessions (
    thread_id      TEXT PRIMARY KEY,
    title          TEXT NOT NULL DEFAULT 'New Chat',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    message_count  INTEGER NOT NULL DEFAULT 0
);
```

Key functions:

- **`create_chat()`** — generates a fresh `uuid4()` as the `thread_id` and
  inserts a row. This same UUID becomes the `thread_id` LangGraph uses to
  key its own checkpoint tables — one ID, shared meaning, two tables.
- **`list_chats()`** — powers the sidebar, ordered by `updated_at DESC` so
  the most recently active chat floats to the top (like ChatGPT's sidebar).
- **`touch_chat(thread_id, message_count)`** — called after every model
  reply to bump `updated_at` and record how big the conversation has grown.
- **`delete_chat(thread_id)`** — this is the one place where we reach
  directly into LangGraph's own tables:

```python
conn.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (thread_id,))
conn.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (thread_id,))
conn.execute("DELETE FROM checkpoints WHERE thread_id = %s", (thread_id,))
conn.execute("DELETE FROM chat_sessions WHERE thread_id = %s", (thread_id,))
```

`PostgresSaver` has no built-in "delete this thread" cascade, so deleting a
chat from the UI means manually clearing all four tables. This is a good
example of why knowing LangGraph's actual schema (not just its API) matters.

---

### `Chatbot/graph.py` — the LangGraph definition itself

The whole graph is deliberately tiny — one node, one edge:

```python
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
```

`MessagesState` is a prebuilt LangGraph state schema with one field,
`messages`, that uses the `add_messages` **reducer**. A reducer defines how
new data merges into existing state. `add_messages` means: normally new
messages are *appended*, not overwritten — that's the mechanic that makes
"memory" work at all. Without it, every `.invoke()` would wipe prior turns.

**Why the connection is opened manually instead of `with ... as checkpointer:`**

LangGraph's own docs show:
```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    ...
```
That closes the Postgres connection the instant the `with` block ends — fine
for a one-shot script, broken for a long-running Streamlit server. So
`build_graph()` instead does:

```python
conn = Connection.connect(get_conn_string(), autocommit=True)
checkpointer = PostgresSaver(conn)
return builder.compile(checkpointer=checkpointer)
```

...and `Chatbot/app.py` wraps `build_graph()` in `@st.cache_resource` so it
only runs **once** per server process, not once per rerun. This is the same
"cache the expensive resource" pattern as the connection pool.

**Table creation is deliberately *not* done here.** `checkpointer.setup()`
(which creates/migrates `checkpoints`, `checkpoint_blobs`, etc.) lives only
in `init_db.py`, run once by hand. If `graph.py` called `.setup()` on every
import, every Streamlit rerun would re-run schema migrations — wasteful and
unnecessary once the tables exist.

**`regenerate_from_edit()`** — the "edit last message" feature:

```python
removals = [RemoveMessage(id=m.id) for m in messages[last_human_idx:]]
graph.update_state(config, {"messages": removals})
return graph.invoke({"messages": [new_human_message]}, config)
```

`RemoveMessage` is a special sentinel that `add_messages` interprets as "delete
the message with this ID" instead of "append this." So editing a message
doesn't rewrite history in place — it:
1. Finds the last human message and everything after it
2. Emits removal instructions for those messages
3. Applies them via `update_state()` (this itself creates a *new* checkpoint)
4. Invokes a fresh turn with the edited text

Nothing is destroyed at the SQL level — old checkpoint rows remain in
`checkpoints`, linked by `parent_checkpoint_id`. You're just telling the
*current* state, going forward, to act as if the edit happened. You can
literally watch the checkpoint count for a thread keep climbing in the DB
monitor even while you're "editing" rather than adding new messages.

---

### `Chatbot/utils.py` — bridging Streamlit's image upload and LangChain's message format

Ollama's multimodal chat format expects a `HumanMessage` whose `content` is
a **list of content blocks** rather than a plain string, when an image is
involved:

```python
content = [
    {"type": "text", "text": text or "Describe this image."},
    {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
]
```

`build_human_message()` handles both cases (text-only vs text+image) so the
rest of the app never has to think about the distinction. `message_text()`
and `message_has_image()` do the reverse — safely pulling display text or
detecting an image out of a message, regardless of which content shape it
has. This matters because once a conversation is loaded back from Postgres,
old messages might be either shape.

**Important caveat documented in the code itself:** only vision-capable
Ollama models (`qwen2.5vl`, `llava`, etc.) actually look at the image block.
A text-only Qwen model will typically ignore it or error — but the image is
still correctly saved into the Postgres checkpoint either way, since it's
just part of the message content LangGraph serializes.

---

### `Chatbot/app.py` — tying it all together in the UI

Walking through what happens on a page load / interaction:

1. **`init_chat_sessions_table()`** runs (idempotent — safe every time).
2. **`get_graph()`** is called through `@st.cache_resource` — builds the
   graph + opens the Postgres connection *once*, reused across reruns.
3. **`st.session_state.active_thread_id`** tracks which chat is currently
   open. On first load, it picks the most recent chat or creates a new one.
4. **Sidebar loop** — for every row in `list_chats()`, renders either the
   chat button, or (if you clicked ✏️) a rename text box, or handles 🗑️
   delete. Each click ends in `st.rerun()`, because Streamlit has no partial
   re-render — the whole script re-executes top to bottom after any state
   change.
5. **Main area** — loads `graph.get_state(config)` for the active thread,
   which pulls the full message history straight out of Postgres, and
   renders it with `st.chat_message()`.
6. **Edit expander** — only shown if there's at least one human message;
   wraps `regenerate_from_edit()`.
7. **Input row** — `st.file_uploader` for an optional image, `st.chat_input`
   for text. On submit: builds the message via `build_human_message()`,
   calls `graph.invoke(...)`, displays the reply, then calls `touch_chat()`
   to update the sidebar's `updated_at` / `message_count`, then reruns.
8. **Auto-titling** — the very first message of a new chat becomes its title
   (truncated to 40 chars), so you don't have to manually rename every chat.

---

### `DB/monitor_app.py` — watching Postgres do its job

A second, independent Streamlit app (run separately: `streamlit run
DB/monitor_app.py`) that queries Postgres system catalogs, not application
tables, to show:

- **Database overview** — total size, Postgres version, live connection
  count (`pg_database_size`, `pg_stat_activity`).
- **Table sizes** — `pg_total_relation_size()` per table, so you can watch
  `checkpoint_blobs` grow as conversations (and any attached images) pile up.
- **Chat sessions** — a raw dump of your `chat_sessions` table.
- **Checkpoints per thread** — `COUNT(*) GROUP BY thread_id` against the
  `checkpoints` table — this is the direct, visual proof of the version
  chain building up as you chat or edit messages.
- **Store table** — currently empty; this is where Phase 3's long-term
  memory will start showing rows.
- **Live activity** — `pg_stat_activity`, useful for confirming the
  connection pool isn't leaking connections.
- **Raw SQL box** — guarded to only allow `SELECT`, for ad-hoc exploration.

Running the chatbot and the monitor side by side is the fastest way to
build real intuition: send a message, switch tabs, watch `checkpoints` and
`checkpoint_blobs` grow in real time.

---

### `init_db.py` — the one-time setup script

```python
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()

init_chat_sessions_table()
```

This is the *only* place `.setup()` is called. It's safe to re-run any
time — LangGraph tracks applied migrations in `checkpoint_migrations` and
`store_migrations`, so re-running just no-ops if everything's current. The
`PostgresStore.setup()` call creates tables Phase 1 doesn't use yet (`store`,
`store_migrations`) — that's intentional groundwork for Phase 2/3.

---

## 3. What you should be able to see in Postgres right now

```sql
\c agent_memory
\dt
```
Should list: `checkpoints`, `checkpoint_blobs`, `checkpoint_writes`,
`checkpoint_migrations`, `store`, `store_migrations`, `chat_sessions`.

```sql
SELECT thread_id, checkpoint_id, parent_checkpoint_id
FROM checkpoints
ORDER BY checkpoint_id;
```
Should show a growing chain per thread — each row's `parent_checkpoint_id`
pointing at the previous checkpoint for that thread. That chain *is*
short-term memory, made physical.

```sql
SELECT * FROM chat_sessions ORDER BY updated_at DESC;
```
Should mirror exactly what the sidebar shows you.

---

## 4. What Phase 1 deliberately does NOT do

- No semantic search / embeddings / vector similarity
- No memory that survives across different `thread_id`s (that's the
  `store` table — created, but unused until Phase 3)
- No automatic summarization or context trimming when a conversation gets
  very long (that's Phase 2)

These aren't missing by accident — Phase 1's entire purpose was to make the
checkpointer mechanism (and Postgres's role in it) fully legible before
adding retrieval or cross-session memory on top.

---

➡️ Continue to [phase_2_explain.md](./phase_2_explain.md) for context-window
management (summarization/trimming) and, after that, Phase 3's cross-thread
long-term memory using the `store` table.