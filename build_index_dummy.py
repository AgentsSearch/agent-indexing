import json
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer

def load_and_index(json_file='mcpagents.json', db_file='agents_mock.db', index_file='agents.index'):
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    index = faiss.IndexFlatIP(384)

    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        agents_data = json.load(f)

    db_data, descriptions = [], []
    for agent in agents_data:
        db_data.append((
            agent.get("agent_id", "unknown"),
            agent.get("name", "Unknown"),
            agent.get("source", "mcp"),
            agent.get("description", ""),
            ", ".join(agent.get("tools", [])),
            ", ".join(agent.get("detected_capabilities", [])),
            agent.get("arena_elo"),
            agent.get("community_rating"),
            agent.get("testability_tier", "UNTESTED"),
            agent.get("pricing", "unknown")
        ))
        descriptions.append(agent.get("description", ""))

    print("Generating embeddings...")
    embeddings = model.encode(descriptions).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_file)

    print("Saving to database...")
    with sqlite3.connect(db_file) as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO agents 
            (agent_id, name, source, description, tools, capabilities, arena_elo, community_rating, testability_tier, pricing) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, db_data)
    print(f"Success! {len(db_data)} agents indexed.")

# This lets you run the file directly OR import the function elsewhere
if __name__ == "__main__":
    load_and_index()