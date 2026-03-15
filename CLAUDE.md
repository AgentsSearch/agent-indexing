# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semantic search and indexing system for MCP (Model Context Protocol) agents. Uses sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim) with FAISS vector search, backed by SQLite for metadata storage. Exposes results via a FastAPI endpoint and integrates with an external probing pipeline for agent evaluation.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline: init DB → build index → test search
python main.py

# Build index from specific JSON files (supports multiple)
python build_index.py --json path/to/agents.json
python build_index.py --json file1.json file2.json

# Run the search API server
python -m uvicorn api:app --reload

# Test search via curl
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a tool to track expenses"}'

# Run probing pipeline (requires external src/ modules and LLM API key)
export CEREBRAS_API_KEY=your_key
python run_pipeline.py --query "Find the current weather in London"
```

No linting, formatting, or test framework is configured.

## Architecture

**Data flow:** JSON agent files → `build_index.py` (embed + index) → `agents.db` + `agents.index` → `api.py` (search) → `run_pipeline.py` (optional evaluation)

### Key Files

- **`main.py`** — Orchestrator: runs `init_db` → `build_index` → `test_search` in sequence
- **`build_index.py`** — Core indexer. Parses agent JSON, normalizes tools/capabilities across different schemas (MCP vs TAAFT), generates embeddings, builds FAISS IndexFlatIP, writes SQLite. Entry: `load_and_index(json_files, db_file, index_file)`
- **`api.py`** — FastAPI server. `POST /search` endpoint encodes query, searches FAISS, applies domain-based score boosting (+0.15 for finance/coding/web keyword matches), returns ranked results from SQLite
- **`init_db.py`** — Creates SQLite `agents` table with indexes on source, pricing, arena_elo
- **`search.py`** — Simple CLI search utility (`test_search()`) for quick verification
- **`run_pipeline.py`** — Probing pipeline integration. Calls the `/search` API, then uses an LLM to decompose queries into subtask DAGs, align to agent tools, generate/validate probes, and score agents. Imports from external `src/` package (not in this repo)

### Data Formats

The system handles two agent record formats in the same index:
- **MCP records**: native format with `agent_id`, `name`, `description`, `tools`, `mcp_server_url`
- **TAAFT records**: converted format with `detected_capabilities` (list or CSV string)

Tools can appear as strings, dicts (with `name` key), or objects — `build_index.py` normalizes all formats. In SQLite, tools and capabilities are stored as CSV strings.

### Generated Artifacts (gitignored)

- `agents.db` — SQLite database with agent metadata
- `agents.index` — Serialized FAISS index
