"""
Chatbot/graph.py

The actual LangGraph definition: one node that calls a local Ollama model.
Conversation state (the message list) is persisted via PostgresSaver,
keyed by thread_id - this is "Phase 1: short-term / thread-scoped memory".

No RAG, no embeddings, no long-term store here on purpose - this file is
meant to teach the checkpointer mechanism in isolation.
"""
import os
from dotenv import load_dotenv
from psycopg import Connection

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_ollama import ChatOllama
from langchain_core.messages import RemoveMessage

from DB.connection import get_conn_string

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.4)


def call_model(state: MessagesState):
    """The only node in this graph: send the full message history to the
    model and append its reply. LangGraph's `add_messages` reducer (built
    into MessagesState) takes care of appending rather than overwriting."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def build_graph():
    """
    Builds and compiles the graph with a live PostgresSaver checkpointer.

    Note: PostgresSaver.from_conn_string(...) in the LangGraph docs is
    shown as a context manager (`with ... as checkpointer:`), which closes
    the connection when the block exits. That's wrong for a long-running
    Streamlit app - we want the connection to stay open for the app's
    lifetime, so we open a plain psycopg Connection ourselves and hand it
    to PostgresSaver directly. Table creation (`.setup()`) is handled once,
    separately, in init_db.py - not here, so we don't re-run migrations on
    every Streamlit rerun.
    """
    builder = StateGraph(MessagesState)
    builder.add_node("model", call_model)
    builder.add_edge(START, "model")

    conn = Connection.connect(get_conn_string(), autocommit=True)
    checkpointer = PostgresSaver(conn)

    return builder.compile(checkpointer=checkpointer)


def regenerate_from_edit(graph, config, new_human_message):
    """
    Implements "edit last message and regenerate".

    LangGraph's checkpointer is a version chain (each turn is a new
    checkpoint pointing at the previous one - see `parent_checkpoint_id`
    in the `checkpoints` table). To edit history, we don't rewrite old
    rows; instead we tell the *current* state to drop the last human
    turn and its reply using RemoveMessage (a special reducer signal
    understood by MessagesState's add_messages function), then invoke a
    fresh turn on top. This creates new checkpoints - the old ones are
    still there in the table if you ever want to inspect the chain.
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