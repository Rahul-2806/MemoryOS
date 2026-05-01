import sqlite3

conn = sqlite3.connect("C:/PROJECT/MemoryOS/data/memory.db")
conn.execute("DELETE FROM memories WHERE session_id IN ('user_001','user_002','user_003')")
conn.commit()
conn.close()
print("Cleared benchmark sessions")