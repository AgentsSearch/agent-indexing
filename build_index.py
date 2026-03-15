import json, sqlite3, faiss, os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
from typing import Iterable
from sentence_transformers import SentenceTransformer

# JSON fields that get serialized as JSON strings in SQLite
_JSON_FIELDS = ("tools", "detected_capabilities", "remotes", "documentation",
                "documentation_chunks", "llm_extracted")


def _tool_names(raw_tools):
    """Extract readable tool names for search text."""
    if not raw_tools:
        return []
    names = []
    for item in raw_tools:
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("tool_name") or item.get("name") or "").strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    return names


def _tool_descriptions(raw_tools):
    """Extract tool descriptions for search text."""
    if not raw_tools:
        return []
    descs = []
    for item in raw_tools:
        if isinstance(item, dict):
            desc = str(item.get("description") or "").strip()
            if desc:
                descs.append(desc)
    return descs


def _build_search_text(agent, description):
    """Combine description, tool names, capabilities, and detected_capabilities
    into a single text for embedding.

    Priority ordering (model truncates at ~256 tokens, attention decays):
      1. description           — always present, clearest intent
      2. tool names (cap 10)   — exact vocabulary: "web_search", "sql_query"
      3. capabilities          — what the agent does
      4. detected_capabilities — lightweight domain tags

    Excluded: limitations (anti-signal), requirements (noise),
    tool descriptions (redundant with capabilities), documentation (markdown noise).
    """
    parts = [description]

    tool_names = _tool_names(agent.get("tools"))[:10]
    if tool_names:
        parts.append("Tools: " + ", ".join(tool_names))

    llm_extracted = agent.get("llm_extracted")
    if isinstance(llm_extracted, dict):
        capabilities = llm_extracted.get("capabilities") or []
        if isinstance(capabilities, list):
            parts.extend(str(item) for item in capabilities if str(item).strip())

    detected = agent.get("detected_capabilities")
    if detected:
        if isinstance(detected, list):
            parts.append(", ".join(str(c) for c in detected))
        elif isinstance(detected, str):
            parts.append(detected)

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


def _json_or_none(value):
    """Serialize a value as JSON string, or None if falsy."""
    return json.dumps(value) if value else None


def _bool_to_int(value):
    """Convert bool/None to SQLite integer (1/0/None)."""
    if value is None:
        return None
    return 1 if value else 0


_CREATE_TABLE = """
CREATE TABLE agents (
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
)
"""

_INSERT = "INSERT INTO agents VALUES ({})".format(", ".join("?" * 32))


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

    db_data, search_texts, seen_ids = [], [], set()

    faiss_id = 0
    for agent in _iter_agent_records(json_files):
        a_id = agent.get("agent_id", "unknown")
        if a_id in seen_ids:
            continue
        seen_ids.add(a_id)

        description = str(agent.get("description", "") or "")
        search_text = _build_search_text(agent, description)

        db_data.append((
            faiss_id,
            a_id,
            agent.get("name", "Unknown"),
            agent.get("source", "mcp"),
            agent.get("source_url"),
            description,
            _json_or_none(agent.get("tools")),
            _json_or_none(agent.get("detected_capabilities")),
            agent.get("llm_backbone"),
            agent.get("arena_elo"),
            agent.get("arena_battles"),
            agent.get("community_rating"),
            agent.get("rating_count"),
            agent.get("pricing", "unknown"),
            agent.get("last_updated"),
            agent.get("indexed_at"),
            agent.get("testability_tier", "UNTESTED"),
            _bool_to_int(agent.get("is_available")),
            agent.get("availability_status"),
            _bool_to_int(agent.get("is_ai_agent")),
            agent.get("agent_classification"),
            agent.get("classification_rationale"),
            _json_or_none(agent.get("remotes")),
            agent.get("probe_status"),
            agent.get("probed_tool_count"),
            agent.get("smithery_config"),
            _json_or_none(agent.get("documentation")),
            _json_or_none(agent.get("documentation_chunks")),
            agent.get("documentation_quality"),
            agent.get("quality_rationale"),
            agent.get("llm_text_source"),
            _json_or_none(agent.get("llm_extracted")),
        ))
        search_texts.append(search_text)
        faiss_id += 1

    if not search_texts:
        raise ValueError("No valid agent records found in provided JSON files")

    print("Generating embeddings...")
    embeddings = model.encode(search_texts, batch_size=16, show_progress_bar=True).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, index_file)

    print("Saving to database...")
    with sqlite3.connect(db_file) as conn:
        conn.execute(_CREATE_TABLE)
        conn.executemany(_INSERT, db_data)

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
