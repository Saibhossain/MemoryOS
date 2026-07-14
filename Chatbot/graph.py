"""
Chatbot/graph.py

The actual LangGraph definition: one node that calls a local Ollama model.
Conversation state (the message list) is persisted via PostgresSaver,
keyed by thread_id - this is "Phase 1: short-term / thread-scoped memory".
Phase 2 adds a second concern:
what subset of that durable history actually gets sent to the model each
turn, so we don't blow past its context window on a long conversation.

Graph shape:

    START -> model -> [conditional] -> summarize_node -> END
                            |
                            +--> END   (if not triggered)

summarize_node no longer deletes messages from state. Full history
stays in the checkpoint and the UI forever, exactly like Phase 1.
The running summary now lives in a separate `summary_context` table
(DB/summary_context.py). call_model reads that table to decide what
to actually send the LLM - bounded input, unbounded storage.


No RAG, no embeddings, no long-term store here on purpose - this file is
meant to teach the checkpointer mechanism in isolation.
"""
import os
import base64
from dotenv import load_dotenv
from psycopg import Connection

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import RemoveMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from Chatbot.memory_tools import build_memory_tools, notes_as_context_with_ids

from DB.profiles import DEFAULT_PROFILE_ID
from DB.connection import get_conn_string
from DB.summary_context import get_summary_context, upsert_summary_context

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.4)

TOKEN_THRESHOLD = 3000  # If the message history exceeds this many tokens, summarize it
MESSAGE_COUNT_FALLBACK = 20 # If we can't get a token count, fall back to this many messages
KEEP_LAST_N = 6

State = MessagesState  # no extra state field needed anymore - summary
                        # lives in Postgres, not in the checkpoint state

def estimate_tokens(messages) -> int:
    total_chars = 0
    for m in messages:
        if isinstance(m.content, str):
            total_chars += len(m.content)

    return total_chars // 4 # Roughly 4 chars per token

def describe_image(image_bytes: bytes, mime: str = "image/png") -> str:
    """
    One-off vision call, deliberately NOT a graph node - it runs before a
    HumanMessage is even constructed, so its output (text) is what gets
    persisted, never the image itself.

    Requires a vision-capable Ollama model (qwen2.5vl, llava, etc). If
    OLLAMA_MODEL is text-only, this will raise - the caller (app.py)
    should catch that and fall back gracefully.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    vision_message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe this image in detail: objects, people, text "
                    "visible in it, colors, and overall context. Be "
                    "concise but thorough - this description will replace "
                    "the image in a text-only conversation log."
                ),
            },
            {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
        ]
    )
    response = llm.invoke([vision_message])
    return response.content


def call_model(state: State, config: RunnableConfig):
    """
    Sends the model only the unsummarized tail of the conversation plus
    the running summary (if any) - both pulled from Postgres, not from
    graph state. `messages` in state itself is always the FULL history;
    we just don't send all of it to the LLM every turn.

    Phase 3 long-term memory notes for the active profile. Tools are
    rebuilt and rebound every call so profile-switching mid-session works
    correctly - see Chatbot/memory_tools.py for why.
    """
    thread_id = config["configurable"]["thread_id"]
    profile_id = config["configurable"].get("profile_id", DEFAULT_PROFILE_ID)
    
    summary, summarized_count = get_summary_context(thread_id)
    memory_context = notes_as_context_with_ids(profile_id)

    all_messages = state["messages"]
    tail_messages = all_messages[summarized_count:]

    system_parts = []
    if memory_context:
        system_parts.append(memory_context)
    if summary:
        system_parts.append(f"Here is a summary of earlier parts of this conversation:\n\n{summary}\n\n Use it for context, but respond only to the most recent message.")

    model_input = tail_messages
    if system_parts:
        system_msg = SystemMessage(content="\n\n---\n\n".join(system_parts))
        model_input = [system_msg] + tail_messages

    tools = build_memory_tools(profile_id, thread_id)
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(model_input)

    return {"messages": [response]}


def tool_node(state: State, config: RunnableConfig):
    """
    Executes whatever tool call(s) the model just made. Not LangGraph's
    prebuilt ToolNode, because our tools need per-turn profile_id/thread_id
    context (see Chatbot/memory_tools.py's factory pattern) rather than a
    fixed tool list bound once at graph-build time.
    """
    thread_id = config["configurable"]["thread_id"]
    profile_id = config["configurable"].get("profile_id", DEFAULT_PROFILE_ID)

    tools = build_memory_tools(profile_id, thread_id)
    tools_by_name = {t.name: t for t in tools}

    last_message = state["messages"][-1]
    tool_messages = []
    for call in last_message.tool_calls:
        tool_fn = tools_by_name.get(call["name"])
        if tool_fn is None:
            result = f"Unknown tool: {call['name']}"
        else:
            result = tool_fn.invoke(call["args"])
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"messages": tool_messages}


def should_summarize(state: State, config: RunnableConfig) -> str:
    """Looks only at the *unsummarized tail* (not the whole history) to
    decide whether there's enough new material to fold into the summary."""
    thread_id = config["configurable"]["thread_id"]
    _, summarized_count = get_summary_context(thread_id)

    unsummarized = state["messages"][summarized_count:]
    if len(unsummarized) <= KEEP_LAST_N:
        return END

    foldable = unsummarized[:-KEEP_LAST_N]
    if not foldable:
        return END

    token_count = estimate_tokens(foldable)
    if token_count > TOKEN_THRESHOLD or len(unsummarized) > MESSAGE_COUNT_FALLBACK:
        return "summarize"

    return END


def route_after_agent(state: State, config: RunnableConfig) -> str:
    """
    Single routing decision after the agent node runs: if it just made
    tool calls, go execute them (and loop back to the agent afterward -
    see build_graph). Otherwise, fall through to the existing Phase 2
    summarization check.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return should_summarize(state, config)


def summarize_node(state: State, config: RunnableConfig):
    """
    Folds newly-eligible older messages into the running summary and
    advances `summarized_count` in Postgres. Does NOT touch `messages` in
    state - nothing is ever removed from the checkpoint or the UI.
    """
    thread_id = config["configurable"]["thread_id"]
    existing_summary, summarized_count = get_summary_context(thread_id)

    all_messages = state["messages"]
    unsummarized = all_messages[summarized_count:]
    foldable = unsummarized[:-KEEP_LAST_N]

    if not foldable:
        return {}

    conversation_text = "\n".join(
        f"{m.type.upper()}: {m.content if isinstance(m.content, str) else '[non-text content]'}"
        for m in foldable
    )

    if existing_summary:
        prompt = (
            "Update the running summary below with the new conversation "
            "excerpt that follows. Keep it concise but preserve important "
            "facts, decisions, and context.\n\n"
            f"EXISTING SUMMARY:\n{existing_summary}\n\n"
            f"NEW EXCERPT TO FOLD IN:\n{conversation_text}\n\n"
            "Return only the updated summary text, nothing else."
        )
    else:
        prompt = (
            "Summarize the following conversation excerpt concisely, "
            "preserving important facts, decisions, and context.\n\n"
            f"{conversation_text}\n\n"
            "Return only the summary text, nothing else."
        )

    new_summary = llm.invoke(prompt).content
    new_summarized_count = summarized_count + len(foldable)

    upsert_summary_context(thread_id, new_summary, new_summarized_count)

    return {}  # state itself is untouched - messages stay complete




def build_graph():
    """
    Builds and compiles the Phase 2 graph with a live PostgresSaver
    checkpointer. Same manual-connection pattern as Phase 1: we keep the
    connection open for the app's lifetime rather than using the
    `with ... as checkpointer:` pattern from LangGraph's docs, which
    would close it immediately - wrong for a long-running Streamlit app.
    """
    builder = StateGraph(State)
    builder.add_node("model", call_model)
    builder.add_node("tools", tool_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", 
        route_after_agent,
        {"tools": "tools", "summarize": "summarize", END: END},
    )
    builder.add_edge("tools", "model")
    builder.add_edge("summarize", END)

    conn = Connection.connect(get_conn_string(), autocommit=True)
    checkpointer = PostgresSaver(conn)

    app = builder.compile(checkpointer=checkpointer)
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("img/Phase3_graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception as e:
        print(f"Error occurreds while generating graph: {e}")

    return app


def regenerate_from_edit(graph, config, new_human_message):
    """Edit-last-message mechanic, unchanged from before. This still uses
    RemoveMessage - that's fine, it's an intentional user-initiated edit
    of the most recent turn, not automatic history pruning."""
    state = graph.get_state(config)
    messages = state.values.get("messages", [])

    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].type == "human":
            last_human_idx = i
            break

    if last_human_idx is not None:
        removals = [RemoveMessage(id=m.id) for m in messages[last_human_idx:]]
        graph.update_state(config, {"messages": removals})

    return graph.invoke({"messages": [new_human_message]}, config)