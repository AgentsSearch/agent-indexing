import json, sqlite3, faiss, os
from sentence_transformers import SentenceTransformer

def load_and_index(json_file='../Agent-Search-Engine/mcp_agents.json', db_file='agents.db', index_file='agents.index'):
    
    if os.path.exists(db_file): os.remove(db_file)
    if os.path.exists(index_file): os.remove(index_file)

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    index = faiss.IndexFlatIP(384)

    with open(json_file, 'r') as f:
        agents_data = json.load(f)

    db_data, descriptions, seen_ids = [], [], set()

    faiss_id = 0
    for agent in agents_data:
        a_id = agent.get("agent_id", "unknown")
        if a_id in seen_ids:
            continue
        seen_ids.add(a_id)

        # Added faiss_id here
        db_data.append((
            faiss_id, a_id, agent.get("name", "Unknown"), agent.get("source", "mcp"),
            agent.get("description", ""), ", ".join(agent.get("tools", [])),
            ", ".join(agent.get("detected_capabilities", [])),
            agent.get("arena_elo"), agent.get("community_rating"),
            agent.get("testability_tier", "UNTESTED"), agent.get("pricing", "unknown")
        ))
        descriptions.append(agent.get("description", ""))
        faiss_id += 1

    print("Generating embeddings...")
    embeddings = model.encode(descriptions).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_file)

    print("Saving to database...")
    with sqlite3.connect(db_file) as conn:
        # Added faiss_id as PRIMARY KEY
        conn.execute("""
            CREATE TABLE agents (
                faiss_id INTEGER PRIMARY KEY,
                agent_id TEXT, name TEXT, source TEXT, description TEXT, 
                tools TEXT, capabilities TEXT, arena_elo REAL, community_rating REAL, 
                testability_tier TEXT, pricing TEXT
            )""")
        conn.executemany("INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", db_data)
        
    print(f"Success! Indexed {len(db_data)} unique agents.")

if __name__ == "__main__":
    load_and_index()