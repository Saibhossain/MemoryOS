# Phase 2 — Context Window Management (Complete)

**Status:** implemented. This replaces the earlier placeholder.

⬅️ Back to [phase_1_explain.md](./phase_1_explain.md)

---

## 1. The problem Phase 2 solves

Phase 1 gave every conversation *perfect, unlimited* memory — the
checkpointer stores every message forever. That's great for durability, but
it creates a real problem: your Qwen model's context window is finite, and
feeding it the entire history of a long conversation every turn will
eventually blow past that limit, slow inference down, and waste tokens.

Phase 2 is about the engineering trade-off between:
- **Checkpointer memory** (full fidelity, replayable, already built in Phase 1)
- **What actually gets sent to the model** on each turn (must be bounded)

![Phase 2 Graph](/img/Phase2_graph.png)

The diagram above is the actual shape of the Phase 2 graph: `model` runs
every turn, a conditional edge decides whether enough new material has
piled up, and if so `summarize_node` runs before returning. Note the three
Postgres tables at the bottom and which node touches which — that mapping
is the core of everything below.

---

## 2. A false start worth understanding (why the design looks the way it does)

The first version of Phase 2 made `summarize_node` delete old messages
from `state["messages"]` using `RemoveMessage`, the same mechanic used for
the "edit last message" feature. This seemed reasonable — "summarized
messages are no longer needed verbatim" — but it caused two real bugs:

1. **Old messages disappeared from the UI.** Once summarized, a
   conversation's earlier turns were gone — not just from the model's
   input, but from the chat history shown in Streamlit and from the
   checkpoint's current state entirely. Re-rendering a chat that had been
   summarized could throw errors when code expected messages that no
   longer existed.
2. **No image data was ever being stored as text**, which is a separate
   issue but was discovered at the same time — the original `utils.py`
   stored raw base64 image bytes inside message content, bloating
   `checkpoint_blobs` for no benefit once the model had already "seen" the
   image once.

The fix for both bugs follows the same underlying principle: **separate
"what's true and complete" (must never be destroyed) from "what's currently
relevant to send the model" (can be bounded/computed on the fly).**

---

## 3. File-by-file: what changed and why

### `DB/summary_context.py` — NEW FILE

A dedicated table, decoupled entirely from LangGraph's checkpoint state:

```sql
CREATE TABLE summary_context (
    thread_id         TEXT PRIMARY KEY REFERENCES chat_sessions(thread_id) ON DELETE CASCADE,
    summary           TEXT NOT NULL DEFAULT '',
    summarized_count  INTEGER NOT NULL DEFAULT 0,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

- **`summary`** — the current running text summary of older parts of the conversation
- **`summarized_count`** — how many messages (counting from index 0) have already been folded into that summary

This is the single most important design change in Phase 2: the summary is
no longer a field inside the LangGraph checkpoint. It's a completely
separate, ordinary SQL table that the app reads and writes directly. This
means:
- `messages` in the checkpoint is **never trimmed** — Phase 1's guarantee
  ("everything is durable, nothing is lost") stays true in Phase 2
- The UI can always render the full conversation, unconditionally
- The `ON DELETE CASCADE` means deleting a chat automatically cleans up its
  summary row too

**Outcome:** old conversations remain fully visible after summarization —
the bug from the false start is gone by construction, not by a patch.

---

### `Chatbot/graph.py` — rewritten

**State** no longer carries a `summary` field:
```python
State = MessagesState  # summary lives in Postgres now, not in checkpoint state
```

**`call_model`** now reads `summary_context` directly (via `get_summary_context`)
instead of `state["summary"]`, and only sends the model the *unsummarized
tail* of the conversation:

```python
thread_id = config["configurable"]["thread_id"]
summary, summarized_count = get_summary_context(thread_id)

all_messages = state["messages"]          # FULL history, untouched
tail_messages = all_messages[summarized_count:]   # only what hasn't been folded in yet

if summary:
    model_input = [SystemMessage(content=f"...summary...\n{summary}")] + tail_messages
else:
    model_input = tail_messages
```

`all_messages` is always complete. Only `tail_messages` — the slice actually
sent to the LLM — is bounded.

**`should_summarize`** (the conditional edge) now looks only at the
*unsummarized* portion when deciding whether to trigger:
```python
unsummarized = state["messages"][summarized_count:]
if len(unsummarized) <= KEEP_LAST_N:
    return END
foldable = unsummarized[:-KEEP_LAST_N]
...
```
`KEEP_LAST_N = 6` (3 exchanges) always stay untouched, exactly as planned.

**`summarize_node`** writes to Postgres, not to graph state:
```python
new_summarized_count = summarized_count + len(foldable)
upsert_summary_context(thread_id, new_summary, new_summarized_count)
return {}   # state is untouched — no RemoveMessage, nothing pruned
```
That `return {}` is the fix. `summarize_node` produces a side effect (a SQL
write) rather than a state mutation. LangGraph is fine with a node that
returns no state delta — it just means "nothing changed in the checkpoint
this step," which is exactly what we want.

**New function: `describe_image()`** — a standalone vision call, deliberately
**not** a graph node:
```python
def describe_image(image_bytes: bytes, mime: str = "image/png") -> str:
    ...
    response = llm.invoke([vision_message])
    return response.content
```
This runs *before* a `HumanMessage` is even constructed. Its output (plain
text) is what gets persisted — the image bytes themselves are only ever
held in memory for the duration of this one call, never written to Postgres.

**Outcome:** the model's input is bounded (context window safe), but
storage and UI are never bounded — the two concerns are fully decoupled.

---

### `Chatbot/utils.py` — rewritten

`build_human_message()` no longer accepts an uploaded image file. It now
accepts an already-generated **text description**:
```python
def build_human_message(text: str, image_description: str | None = None) -> HumanMessage:
    if image_description is None:
        return HumanMessage(content=text)
    combined = f"{text}\n\n[Attached image - vision analysis: {image_description}]"
    return HumanMessage(content=combined)
```
Every message is now a plain string — no multimodal content-block lists are
stored anymore. `message_text()` stays defensive (handles old-format
messages if any exist from before this change), and a new helper
`message_has_image_note()` replaces the old `message_has_image()`, detecting
the `[Attached image...]` marker in text instead of inspecting content-block
structure.

**Outcome:** `checkpoint_blobs` never grows from image uploads — an image
becomes a paragraph of text, and text is cheap to store and cheap for the
summarizer to fold into a summary later, exactly like any other message.

---

### `Chatbot/app.py` — updated

Three changes tie the above together:

1. **Init:** `init_summary_context_table()` added alongside
   `init_chat_sessions_table()` — both idempotent, both run on every app start.
2. **Image upload flow:**
   ```python
   if uploaded_image is not None:
       with st.spinner("👁️ Analyzing image..."):
           image_description = describe_image(image_bytes, mime)
   human_msg = build_human_message(user_text, image_description)
   ```
   Wrapped in try/except — if `OLLAMA_MODEL` isn't vision-capable, the app
   warns and falls back to text-only rather than crashing.
3. **Summary panel + sidebar badge:** the summary is now read via
   `get_summary_context(thread_id)` (a plain SQL lookup) instead of
   `graph.get_state(...).values.get("summary")` — fast, and correct, since
   that field doesn't live in graph state anymore.

**Outcome:** the chat UI always shows every message, always. Summarization
is visible (🧵 spinner, `st.info()` banner, collapsible summary panel) but
never destructive.

---

### `init_db.py` — updated

One addition:
```python
from DB.summary_context import init_summary_context_table
...
init_summary_context_table()
print("✅ summary_context table ready")
```
Run once (`python init_db.py`) any time after pulling these changes — it's
idempotent, so it's always safe to re-run.

---

### `DB/monitor_app.py` — fully redesigned

This went from a single scrolling page to a proper multi-page dashboard
with sidebar navigation. Two things changed for correctness, and the rest
is new functionality:

**Bug fix:** the old dashboard tried to read the summary via
`graph.get_state(config).values.get("summary", "")` — that field no longer
exists in graph state. It now calls `get_summary_context(thread_id)` from
`DB/summary_context.py`, the same function the chatbot itself uses.

**New pages:**

| Page | What it shows |
|---|---|
| 📊 Overview | DB size, version, connections, total chats/checkpoints, table size bar + pie, row counts, top-20 threads by checkpoint count |
| 🧵 Threads | Searchable chat list **+ per-thread deep dive**: summary panel, checkpoint chain diagram, role pie chart, per-turn length bar chart, cumulative-growth area chart with a marked summarization boundary, full expandable message log |
| 💾 Storage & Health | Table/index size breakdown, dead-tuple/vacuum stats, a working **VACUUM ANALYZE** button |
| 🔌 Live Activity | `pg_stat_activity` view + a working **terminate connection** control |
| 🛠️ SQL Console | Read-only by default; an explicit "write mode" checkbox unlocks INSERT/UPDATE/DELETE/DDL with a visible warning |
| 🧠 Long-term Memory | `store` table browser with namespace filtering — ready for Phase 3, empty until then |

**Outcome:** you can now watch, for any single conversation, exactly how
many messages exist vs. how many the model actually sees, when
summarization kicked in, and what the summary currently says — all without
touching `psql`.

---

## 4. How to verify all of this actually works

1. Run `python init_db.py` once (creates `summary_context` if it's missing).
2. Start a long conversation in `streamlit run Chatbot/app.py` — enough
   messages to cross `MESSAGE_COUNT_FALLBACK` (20) or a low
   `TOKEN_THRESHOLD` if you've temporarily reduced it for testing.
3. Watch for the 🧵 "Compacting older messages..." spinner state and the
   `st.info()` banner after it completes.
4. Confirm every earlier message is **still visible** in the chat — this is
   the regression test for the original bug.
5. Upload an image and send it — confirm the reply references what's in the
   image, and that no `st.image()` preview persists after a page reload
   (only the text description does).
6. Open `streamlit run DB/monitor_app.py` → **🧵 Threads** → pick that
   conversation → confirm:
   - "Total messages (stored)" > "Messages sent to model" (proves trimming
     of *model input* without deletion of *storage*)
   - The summary panel shows real text
   - The cumulative-growth chart shows the orange summarization-boundary line

```sql
-- Direct proof in psql that nothing was deleted:
SELECT thread_id, summary, summarized_count FROM summary_context;
SELECT count(*) FROM checkpoints WHERE thread_id = '<your-thread-id>';
```

---

## 5. What Phase 2 still deliberately does not do

- No semantic search or embeddings — summarization is a plain text-in/text-out
  LLM call, not a retrieval mechanism
- No memory that survives across different `thread_id`s — summaries are
  per-thread, just like checkpoints (that's Phase 3's job, using the
  already-created but still-empty `store` table)
- No token-exact accounting — `estimate_tokens()` uses a `chars / 4`
  heuristic, which is intentionally rough rather than pulling in a
  tokenizer built for a different model family

---
➡️ Continue to [phase_3_explain.md](./phase_3_explain.md). Phase 3 (long-term, cross-thread memory) once this is solid.

