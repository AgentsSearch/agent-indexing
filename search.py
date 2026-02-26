import faiss
import sqlite3
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('agents.index')
k = 5 

def test_search(query):
    vec = model.encode([query]).astype('float32')
    faiss.normalize_L2(vec)
    
    distances, ids = index.search(vec, k) 
    
    conn = sqlite3.connect('agents.db')
    print(f"\nQuery: '{query}'")
    for idx in ids[0]:
        # Search using exact faiss_id instead of rowid
        res = conn.execute("SELECT name, description FROM agents WHERE faiss_id = ?", (int(idx),)).fetchone()
        if res:
            print(f"- {res[0]}: {res[1][:100]}...")
    conn.close()

#test_search("I need a tool to track and split expenses")