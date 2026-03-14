import json
import os
from fastapi import FastAPI, Depends, HTTPException
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

# 1. Domains dictionary needed for the boost
DOMAINS = {
    "finance": ["expense", "money", "split", "bill", "pay"],
    "coding": ["code", "git", "sql", "database", "repository"],
    "web": ["scrape", "browser", "html", "search", "url"]
}

class SearchQuery(BaseModel):
    query: str
    limit: int = 50

@app.post("/search")
def search(req: SearchQuery, _=Depends(verify_token)):
    vec = model.encode([req.query]).astype('float32')
    faiss.normalize_L2(vec)

    distances, ids = index.search(vec, req.limit)
    
    # 2. Defines active_domains here
    q_lower = req.query.lower()
    active_domains = [d for d, words in DOMAINS.items() if any(w in q_lower for w in words)]
    
    results = []
    with sqlite3.connect('agents.db') as conn:
        for i, faiss_id in enumerate(ids[0]):
            res = conn.execute(
                "SELECT agent_id, name, description, tools, capabilities, mcp_server_url, documentation, llm_extracted FROM agents WHERE faiss_id = ?",
                (int(faiss_id),)
            ).fetchone()

            if res:
                base_score = float(distances[0][i])
                agent_text = (res[2] + " " + res[4]).lower()

                # 3. Uses active_domains here
                for domain in active_domains:
                    if any(w in agent_text for w in DOMAINS[domain]):
                        base_score += 0.15
                        break

                results.append({
                    "agent_id": res[0],
                    "name": res[1],
                    "description": res[2],
                    "tools": [t.strip() for t in res[3].split(",") if t.strip()],
                    "capabilities": [c.strip() for c in res[4].split(",") if c.strip()],
                    "mcp_server_url": res[5],
                    "documentation": json.loads(res[6]) if res[6] else None,
                    "llm_extracted": json.loads(res[7]) if res[7] else None,
                    "score": round(base_score, 4)
                })
    
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {"results": results}