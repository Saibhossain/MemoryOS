from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
import os
from dotenv import load_dotenv
load_dotenv()
# Initialize the PostgresStore with the connection string from the .env file

DB_URL = os.getenv("DATABASE_URL")

with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    checkpointer.setup()

with PostgresStore.from_conn_string(DB_URL) as store:
    store.setup()

print("Postgres database setup complete. Table Created")