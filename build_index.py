import json, sqlite3, faiss, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
from typing import Iterable
from sentence_transformers import SentenceTransformer


def _normalize_tools(raw_tools):
    """Normalize tools to a list of readable tool names."""
    if not raw_tools:
        return []

    normalized = []
    for item in raw_tools:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            # MCP servers typically use tool_name; converted records may use name.
            name = str(item.get("tool_name") or item.get("name") or "").strip()
        else:
            name = str(item).strip()

        if name:
            normalized.append(name)
    return normalized


def _normalize_capabilities(agent):
    capabilities = agent.get("detected_capabilities") or []
    if isinstance(capabilities, list):
        return [str(c).strip() for c in capabilities if str(c).strip()]
    if isinstance(capabilities, str):
        return [c.strip() for c in capabilities.split(",") if c.strip()]
    return []


def _build_search_text(agent, description):
    """Combine description, llm_extracted capabilities, and documentation
    into a single text for embedding. Truncate to avoid excessively long inputs."""
    parts = [description]

    llm_extracted = agent.get("llm_extracted")
    if isinstance(llm_extracted, dict):
        for field in ("capabilities", "limitations", "requirements"):
            items = llm_extracted.get(field) or []
            if isinstance(items, list):
                parts.extend(str(item) for item in items if str(item).strip())

    doc = agent.get("documentation")
    if isinstance(doc, dict):
        # Prefer detail_page; fall back to any string value
        doc_text = doc.get("detail_page") or ""
        if not doc_text:
            for v in doc.values():
                if isinstance(v, str) and v.strip():
                    doc_text = v
                    break
        if doc_text:
            # Keep first 500 chars to avoid noise from long pages
            parts.append(doc_text[:500])

    return " ".join(parts)


def _iter_agent_records(json_files: Iterable[str]):
    """Yield agent records across one or more JSON files."""
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for agent in data:
                    if isinstance(agent, dict):
                        yield agent


def _normalize_json_inputs(json_files):
    """Accept either a single path string or a list/tuple of paths."""
    if isinstance(json_files, str):
        return [json_files]
    if isinstance(json_files, (list, tuple)):
        return [str(p) for p in json_files]
    raise TypeError("json_files must be a path string or list/tuple of paths")


def load_and_index(
    json_files='../Agent-Search-Engine/mcp_agents.json',
    db_file='agents.db',
    index_file='agents.index'
):
    json_files = _normalize_json_inputs(json_files)

    for json_file in json_files:
        if not Path(json_file).exists():
            raise FileNotFoundError(f"Input JSON not found: {json_file}")

    if os.path.exists(db_file):
        os.remove(db_file)
    if os.path.exists(index_file):
        os.remove(index_file)

    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    index = faiss.IndexFlatIP(384)

    db_data, descriptions, seen_ids = [], [], set()

    faiss_id = 0
    for agent in _iter_agent_records(json_files):
        a_id = agent.get("agent_id", "unknown")
        if a_id in seen_ids:
            continue
        seen_ids.add(a_id)

        tool_names = _normalize_tools(agent.get("tools", []))
        capabilities = _normalize_capabilities(agent)
        description = str(agent.get("description", "") or "")
        search_text = _build_search_text(agent, description)

        # Serialize rich fields as JSON strings for probing
        documentation = agent.get("documentation")
        doc_json = json.dumps(documentation) if documentation else None
        llm_extracted = agent.get("llm_extracted")
        llm_json = json.dumps(llm_extracted) if llm_extracted else None

        db_data.append((
            faiss_id, a_id, agent.get("name", "Unknown"), agent.get("source", "mcp"),
            description, ", ".join(tool_names),
            ", ".join(capabilities),
            agent.get("arena_elo"), agent.get("community_rating"),
            agent.get("testability_tier", "UNTESTED"), agent.get("pricing", "unknown"),
            agent.get("mcp_server_url", f"http://localhost/{a_id}"),
            doc_json, llm_json
        ))
        descriptions.append(search_text)
        faiss_id += 1

    if not descriptions:
        raise ValueError("No valid agent records found in provided JSON files")

    print("Generating embeddings...")
    embeddings = model.encode(descriptions, batch_size=16, show_progress_bar=True).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_file)

    print("Saving to database...")
    with sqlite3.connect(db_file) as conn:
        conn.execute("""
            CREATE TABLE agents (
                faiss_id INTEGER PRIMARY KEY,
                agent_id TEXT, name TEXT, source TEXT, description TEXT,
                tools TEXT, capabilities TEXT, arena_elo REAL, community_rating REAL,
                testability_tier TEXT, pricing TEXT, mcp_server_url TEXT,
                documentation TEXT, llm_extracted TEXT
            )""")
        conn.executemany("INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", db_data)
        
    print(f"Indexed sources: {', '.join(json_files)}")
    print(f"Success! Indexed {len(db_data)} unique agents.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS + SQLite index from agent JSON files")
    parser.add_argument(
        "--json",
        nargs="+",
        default=["../Agent-Search-Engine/mcp_agents.json", "../TAAFT-Scraping/agents_mcp.json"],
        help="One or more input JSON files (MCP or converted TAAFT)",
    )
    parser.add_argument("--db", default="agents.db", help="Output SQLite DB path")
    parser.add_argument("--index", default="agents.index", help="Output FAISS index path")
    args = parser.parse_args()

    load_and_index(args.json, args.db, args.index)