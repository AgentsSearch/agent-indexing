import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException

load_dotenv()
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import faiss
import sqlite3
from sentence_transformers import SentenceTransformer

app = FastAPI()
security = HTTPBearer()
API_TOKEN = os.environ.get("API_TOKEN")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not API_TOKEN or credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    return credentials

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('agents.index')

DOMAINS = {
    "finance": ["expense", "money", "split", "bill", "pay", "invoice", "budget", "accounting"],
    "coding": ["code", "git", "sql", "database", "repository", "github", "debug", "lint"],
    "web": ["scrape", "browser", "html", "search", "url", "crawl", "fetch", "http"]
}

_JSON_COLUMNS = {"tools", "detected_capabilities", "remotes", "documentation",
                 "documentation_chunks", "llm_extracted"}

class SearchQuery(BaseModel):
    query: str
    limit: int = 50

class KeywordSearchQuery(BaseModel):
    query: str
    limit: int = 50

@app.post("/search")
def search(req: SearchQuery, _=Depends(verify_token)):
    vec = model.encode([req.query]).astype('float32')
    faiss.normalize_L2(vec)

    distances, ids = index.search(vec, req.limit)

    q_lower = req.query.lower()
    active_domains = [d for d, words in DOMAINS.items() if any(w in q_lower for w in words)]

    results = []
    with sqlite3.connect('agents.db') as conn:
        conn.row_factory = sqlite3.Row
        for i, faiss_id in enumerate(ids[0]):
            row = conn.execute(
                "SELECT * FROM agents WHERE faiss_id = ?",
                (int(faiss_id),)
            ).fetchone()

            if row:
                base_score = float(distances[0][i])
                llm_caps = ""
                if row["llm_extracted"]:
                    try:
                        extracted = json.loads(row["llm_extracted"])
                        caps = extracted.get("capabilities") or []
                        if isinstance(caps, list):
                            llm_caps = " ".join(str(c) for c in caps)
                    except (json.JSONDecodeError, TypeError):
                        pass
                agent_text = ((row["description"] or "") + " " + (row["detected_capabilities"] or "") + " " + llm_caps).lower()

                for domain in active_domains:
                    if any(w in agent_text for w in DOMAINS[domain]):
                        base_score += 0.05
                        break

                result = dict(row)
                # Deserialize JSON columns
                for col in _JSON_COLUMNS:
                    if result.get(col):
                        result[col] = json.loads(result[col])
                # Convert SQLite ints back to bools
                for col in ("is_available", "is_ai_agent"):
                    v = result.get(col)
                    result[col] = bool(v) if v is not None else None

                result["score"] = round(base_score, 4)
                del result["faiss_id"]
                results.append(result)

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"results": results}

_KEYWORD_SEARCH_FIELDS = ("name", "description", "tools", "detected_capabilities",
                          "documentation", "llm_extracted")

@app.post("/search/keyword")
def keyword_search(req: KeywordSearchQuery, _=Depends(verify_token)):
    keywords = [k for k in req.query.lower().split() if k]
    if not keywords:
        return {"results": []}

    with sqlite3.connect('agents.db') as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM agents").fetchall()

    results = []
    for row in rows:
        text_blob = " ".join((row[f] or "") for f in _KEYWORD_SEARCH_FIELDS).lower()
        score = sum(1 for kw in keywords if kw in text_blob)
        if score == 0:
            continue

        result = dict(row)
        for col in _JSON_COLUMNS:
            if result.get(col):
                result[col] = json.loads(result[col])
        for col in ("is_available", "is_ai_agent"):
            v = result.get(col)
            result[col] = bool(v) if v is not None else None
        result["score"] = score
        result.pop("faiss_id", None)
        results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    return {"results": results[:req.limit]}
