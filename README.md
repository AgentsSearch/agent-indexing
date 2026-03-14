### Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your values
```

See `.env.example` for required environment variables (`API_TOKEN`, `CEREBRAS_API_KEY`).

### Build DB and Index

Run `main.py` to execute the entire pipeline (init DB, build index, test search):

```bash
python main.py
```

Or build directly from one or more JSON files:

```bash
python build_index.py --json mcp_ai_agents_remote.json
python build_index.py --json file1.json file2.json
```

Notes:
- The indexer accepts multiple files in one run.
- It supports MCP records and converted TAAFT records in the same index.

### Run API

```bash
export API_TOKEN=your-secret-token
python -m uvicorn api:app --reload
```

### Test Search

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"query": "I need a tool to track expenses"}'
```

### For Probing

1. Build the index (see above)
2. Run the API on port 8000
3. Run the probing pipeline:

```bash
export CEREBRAS_API_KEY=your-key
python run_pipeline.py --query "Find the current weather in London"
```
