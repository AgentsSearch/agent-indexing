import sqlite3

# Creates a local file called agents_mock.db
conn = sqlite3.connect('agents.db')
cur = conn.cursor()

# Create table (modified slightly for SQLite compatibility)
def initialize_db():
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    description TEXT,
    tools TEXT,
    capabilities TEXT,
    arena_elo REAL,
    community_rating REAL,
    testability_tier TEXT,
    pricing TEXT,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Create indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_source ON agents (source);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_pricing ON agents (pricing);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_elo ON agents (arena_elo);")

    conn.commit()
    conn.close()

    print("Mock database ready!")