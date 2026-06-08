from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.pipeline import config_a, config_b, config_c

app = FastAPI(title="RAG Hallucination Research API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    config: str = "A"


@app.post("/query")
async def query(req: QueryRequest):
    if req.config == "A":
        return config_a.query(req.question)
    elif req.config == "B":
        return config_b.query(req.question)
    elif req.config == "C":
        return config_c.query(req.question)
    else:
        raise HTTPException(status_code=400, detail="config must be 'A', 'B', or 'C'")


@app.get("/health")
async def health():
    return {"status": "ok", "message": "RAG Hallucination Research API is running"}
