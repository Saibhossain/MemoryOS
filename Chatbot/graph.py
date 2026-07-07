import os
import sys
from typing import Annotated, Sequence, TypedDict

# Ensure proper path routing from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from DB.connection import get_conn_string, get_pool

# 1. State Definition
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 2. Node Execution Block
# Note: Ensure you have pulled the model locally via: ollama pull qwen2.5
llm = ChatOllama(model="qwen3:1.7b", temperature=0.7)

def chatbot_node(state: AgentState):
    """Executes the local model using the structural message state history."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 3. Graph Assembly
workflow = StateGraph(AgentState)
workflow.add_node("chatbot", chatbot_node)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 4. Persistence Initialization & Compiling
DB_URL = get_conn_string()

# Ensure the core LangGraph tables and app metadata tables exist
def init_database_tables():
    # Setup underlying LangGraph storage schemas
    with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
        checkpointer.setup()
        
    # Setup the application-level 'chat_sessions' table required by the monitor
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    message_count INT DEFAULT 0
                );
            """)

init_database_tables()

# Compile graph using the Postgres persistence architecture
# Compile graph using the Postgres persistence architecture via your active pool
pool = get_pool()
checkpointer = PostgresSaver(pool)
app = workflow.compile(checkpointer=checkpointer)