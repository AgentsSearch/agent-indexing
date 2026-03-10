from build_index import load_and_index
from search import test_search 
from init_db import initialize_db
from pathlib import Path

print("Starting pipeline...")

# Initialize the database
initialize_db()

# Run the indexer
input_files = ["../Agent-Search-Engine/mcp_agents.json"]

# Include converted TAAFT dataset if present.
taaft_mcp = Path("../TAAFT-Scraping/agents_mcp.json")
if taaft_mcp.exists():
	input_files.append(str(taaft_mcp))

load_and_index(input_files, "agents.db", "agents.index")

# Test a search instantly
test_search("I need a tool to track and split expenses")