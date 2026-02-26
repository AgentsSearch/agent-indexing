from build_index import load_and_index
from search import test_search 
from init_db import initialize_db

print("Starting pipeline...")

# Initialize the database
initialize_db()

# Run the indexer
load_and_index("../Agent-Search-Engine/mcp_agents.json", "agents.db", "agents.index")

# Test a search instantly
test_search("I need a tool to track and split expenses")