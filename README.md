# Agent Indexing

Semantic search and indexing for MCP agents. Uses sentence-transformer embeddings (all-MiniLM-L6-v2, 384-dim) with FAISS vector search, backed by SQLite for metadata storage. Exposes results via a FastAPI endpoint.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your API_TOKEN
```

## Build Index

Run `main.py` to execute the full pipeline (init DB, build index, test search):

```bash
python main.py
```

Or build directly from one or more JSON files:

```bash
python build_index.py --json mcp_ai_agents.json
python build_index.py --json file1.json file2.json
```

The indexer accepts multiple files in one run and supports both MCP and TAAFT record formats.

## Run API

```bash
export API_TOKEN=your-secret-token
python -m uvicorn api:app --reload
```

## API Contract

### `POST /search`

Semantic search over indexed MCP agents.

#### Authentication

Bearer token via `Authorization` header.

```
Authorization: Bearer <API_TOKEN>
```

#### Request

```jsonc
{
  "query": "string",   // required — natural language search query
  "limit": 50          // optional — max results (default: 50)
}
```

#### Example

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a tool to track expenses", "limit": 10}'
```

#### Response `200`

```jsonc
{
  "results": [
    {
      "agent_id": "string",
      "name": "string",
      "source": "string",                    // e.g. "mcp"
      "source_url": "string | null",
      "description": "string",
      "tools": [                             // array of tool objects | null
        {
          "name": "string",
          "description": "string",
          "inputSchema": {},                 // JSON Schema object
          "annotations": {},                 // optional hints
          "execution": {}                    // optional execution config
        }
      ],
      "detected_capabilities": ["string"],   // array of strings | null
      "llm_backbone": "string | null",
      "arena_elo": "number | null",
      "arena_battles": "number | null",
      "community_rating": "number | null",
      "rating_count": "number | null",
      "pricing": "string | null",
      "last_updated": "string | null",       // ISO timestamp
      "indexed_at": "string | null",         // ISO timestamp
      "testability_tier": "string | null",
      "is_available": "boolean | null",
      "availability_status": "string | null",
      "is_ai_agent": "boolean | null",
      "agent_classification": "string | null",
      "classification_rationale": "string | null",
      "remotes": [{}],                       // array of remote configs | null
      "probe_status": "string | null",
      "probed_tool_count": "number | null",
      "smithery_config": "string | null",
      "documentation": {                     // object | null
        "readme": "string"
      },
      "documentation_chunks": [{}],          // array | null
      "documentation_quality": "number | null",
      "quality_rationale": "string | null",
      "llm_text_source": "string | null",
      "llm_extracted": {                     // object | null
        "capabilities": ["string"],
        "limitations": ["string"],
        "requirements": ["string"]
      },
      "score": 0.8523                        // similarity score, higher = better
    }
  ]
}
```

#### Response `401`

```json
{"detail": "Invalid or missing token"}
```

#### Notes

- Results sorted by `score` descending
- Score is cosine similarity (0–1) with a +0.15 domain boost for finance/coding/web keyword matches
- Score can exceed 1.0 when the domain boost is applied
