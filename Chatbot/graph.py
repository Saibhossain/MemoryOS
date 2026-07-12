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

State now carries a `summary` field alongside `messages`. Both are
persisted in the same Postgres checkpoint - no new tables needed, this
is just one more key in the same JSONB/blob LangGraph already writes.

No RAG, no embeddings, no long-term store here on purpose - this file is
meant to teach the checkpointer mechanism in isolation.
"""
import os
from dotenv import load_dotenv
from psycopg import Connection

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import RemoveMessage, SystemMessage

from DB.connection import get_conn_string

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.4)

TOKEN_THRESHOLD = 3000  # If the message history exceeds this many tokens, summarize it
MESSAGE_COUNT_FALLBACK = 20 # If we can't get a token count, fall back to this many messages
KEEP_LAST_N = 6

class State(MessagesState):
    """
    Subclass of MessagesState that adds a `summary` field to the state.
    This is a simple string, not a list of messages, because we don't
    need to preserve the model's summary turn as a message in the history.
    """
    summary: str

def estimate_tokens(messages) -> int:
    total_chars = 0
    for m in messages:
        if isinstance(m.countent, str):
            total_chars += len(m.content)
        else:
            for block in m.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    total_chars += len(block["text"])
    return total_chars // 4  # Roughly 4 chars per token



def call_model(state: State):
    """The main chat turn. If a summary already exists, it's prepended as
    a SystemMessage so the model keeps continuity even though the raw
    messages it's summarizing have been pruned out of state."""
    messages = state["messages"]
    summary = state.get("summary","")

    if summary:
        system_msg = SystemMessage(
            content=(
             "Here is a summary of the earlier part of this conversation"
             f"that is no longer shown verbatim:\n\n{summary}\n\n"
             "Use it for context, but respond only to the most recent message."
             
            )
        )
        models_input = [system_msg] + messages
    else:
        models_input = messages

    response = llm.invoke(models_input)
    return {"messages": [response]}




def should_summarize(state: State) -> str:
    """Conditional edge: decide whether to route to summarize_node or END."""
    messages = state["messages"]
    if len(messages)<=KEEP_LAST_N:
        return END  # Don't summarize if we have only a few messages
    
    token_count = estimate_tokens(messages)
    if token_count > TOKEN_THRESHOLD or len(messages) > MESSAGE_COUNT_FALLBACK:
        return "summarize"
    
    return END


def summarize_node(state: State):
    """Folds everything older than the last KEEP_LAST_N messages into a
    running summary, then removes those older messages from state using
    RemoveMessage - the same mechanic Phase 1 used for edit/regenerate.

    Nothing is lost from Postgres: the full history is still sitting in
    older checkpoint rows (parent_checkpoint_id chain). We're only
    pruning what gets carried forward in *current* state."""
    messages = state["messages"]
    existing_summary = state.get("summary", "")

    older_messages = messages[:-KEEP_LAST_N]
    if not older_messages:
        return{}
    
    conversation_text = "\n".join(
        f"{m.type.upper()}:{m.content if isinstance(m.content,str) else '[message with image]'}"
        for m in older_messages
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

    summary_response = llm.invoke(prompt)
    new_summary = summary_response.content
    removals = [RemoveMessage(id=m.id) for m in older_messages]

    return {"summary": new_summary, "messages": removals}




def build_graph():
    """
    Builds and compiles the Phase 2 graph with a live PostgresSaver
    checkpointer. Same manual-connection pattern as Phase 1: we keep the
    connection open for the app's lifetime rather than using the
    `with ... as checkpointer:` pattern from LangGraph's docs, which
    would close it immediately - wrong for a long-running Streamlit app.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "model")
    builder.add_conditional_edges(
        "model", 
        should_summarize,
        {"summarize": "summarize", END: END},
    )
    builder.add_edge("summarize", END)

    conn = Connection.connect(get_conn_string(), autocommit=True)
    checkpointer = PostgresSaver(conn)

    app = builder.compile(checkpointer=checkpointer)
    try:
        png_bytes = app.get_graph().draw_mermaid_png()
        with open("img/Phase2_graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception as e:
        print(f"Error occurred while generating graph: {e}")

    return app

def regenerate_from_edit(graph, config, new_human_message):
    """
    Implements "edit last message and regenerate" (unchanged mechanic
    from Phase 1). Note this only removes/replaces the last human turn -
    it does not touch `summary`, so an edit doesn't undo prior
    summarization, only the most recent exchange.
    """
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