import os 
from dotenv import load_dotenv
import psycopg

load_dotenv()
conn = psycopg.connect(os.getenv("DATABASE_URL"))
print(conn.execute("SELECT version();").fetchone())
conn.close()