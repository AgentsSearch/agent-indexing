import sqlite3


def initialize_db(db_file='agents.db'):
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        faiss_id INTEGER PRIMARY KEY,
        agent_id TEXT,
        name TEXT,
        source TEXT,
        source_url TEXT,
        description TEXT,
        tools TEXT,
        detected_capabilities TEXT,
        llm_backbone TEXT,
        arena_elo REAL,
        arena_battles INTEGER,
        community_rating REAL,
        rating_count INTEGER,
        pricing TEXT,
        last_updated TEXT,
        indexed_at TEXT,
        testability_tier TEXT,
        is_available INTEGER,
        availability_status TEXT,
        is_ai_agent INTEGER,
        agent_classification TEXT,
        classification_rationale TEXT,
        remotes TEXT,
        probe_status TEXT,
        probed_tool_count INTEGER,
        smithery_config TEXT,
        documentation TEXT,
        documentation_chunks TEXT,
        documentation_quality REAL,
        quality_rationale TEXT,
        llm_text_source TEXT,
        llm_extracted TEXT
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_source ON agents (source);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_pricing ON agents (pricing);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_elo ON agents (arena_elo);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_available ON agents (is_available);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_agents_classification ON agents (agent_classification);")

    conn.commit()
    conn.close()

    print("Database ready!")
