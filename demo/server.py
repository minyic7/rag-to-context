"""
server.py — FastAPI backend for the RAG workshop demo.

Exposes all 5 RAG approaches as REST endpoints.
Run via Docker:  docker compose up --build
Or directly:     uvicorn server:app --host 0.0.0.0 --port 8000
"""

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="RAG Workshop API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent / "data"


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class ChunkResult(BaseModel):
    url: str
    title: str
    text: str
    score: float
    method: str
    source: Optional[str] = None
    rerank_score: Optional[int] = None
    via: Optional[str] = None


class RetrieveResponse(BaseModel):
    query: str
    mode: str
    chunks: list[ChunkResult]
    elapsed_ms: float


class ToolCall(BaseModel):
    tool: str
    args: dict
    result_preview: str


class AgenticResponse(BaseModel):
    query: str
    mode: str
    answer: str
    tool_log: list[ToolCall]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    available_modes: list[str]
    embeddings_ready: bool
    annual_report_ready: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _has_embeddings() -> bool:
    return (DATA_DIR / "embeddings.npz").exists()


def _has_annual_report() -> bool:
    return (DATA_DIR / "annual_report_text.txt").exists()


def _run_retrieve(mode: str, query: str, top_k: int) -> RetrieveResponse:
    """Dispatch to the correct retriever and wrap the result."""
    if mode == "naive":
        from rag.naive import retrieve
    elif mode == "advanced":
        from rag.advanced import retrieve
    elif mode == "modular":
        from rag.modular import retrieve
    elif mode == "graph":
        from rag.graph_rag import retrieve
    else:
        raise HTTPException(status_code=400, detail=f"Unknown mode: {mode}")

    if mode in ("advanced", "modular", "graph") and not _has_embeddings():
        raise HTTPException(
            status_code=503,
            detail="embeddings.npz not ready. Wait for container startup to finish.",
        )

    t0 = time.time()
    try:
        chunks = retrieve(query, top_k=top_k)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    elapsed_ms = (time.time() - t0) * 1000

    return RetrieveResponse(
        query=query,
        mode=mode,
        chunks=[
            ChunkResult(
                url=c.get("url", ""),
                title=c.get("title", ""),
                text=c.get("text", "")[:500],
                score=c.get("rerank_score", c.get("score", 0)),
                method=c.get("method", mode),
                source=c.get("source"),
                rerank_score=c.get("rerank_score"),
                via=c.get("via"),
            )
            for c in chunks
        ],
        elapsed_ms=round(elapsed_ms, 1),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
def health():
    modes = ["naive"]
    emb = _has_embeddings()
    if emb:
        modes.extend(["advanced", "modular", "graph"])
    modes.append("agentic")
    return HealthResponse(
        status="ok",
        available_modes=modes,
        embeddings_ready=emb,
        annual_report_ready=_has_annual_report(),
    )


@app.post("/api/naive", response_model=RetrieveResponse)
def naive_rag(req: QueryRequest):
    return _run_retrieve("naive", req.query, req.top_k)


@app.post("/api/advanced", response_model=RetrieveResponse)
def advanced_rag(req: QueryRequest):
    return _run_retrieve("advanced", req.query, req.top_k)


@app.post("/api/modular", response_model=RetrieveResponse)
def modular_rag(req: QueryRequest):
    return _run_retrieve("modular", req.query, req.top_k)


@app.post("/api/graph", response_model=RetrieveResponse)
def graph_rag(req: QueryRequest):
    return _run_retrieve("graph", req.query, req.top_k)


@app.post("/api/agentic", response_model=AgenticResponse)
def agentic_rag(req: QueryRequest):
    from rag.agentic import run_agent

    t0 = time.time()
    try:
        answer, tool_log = run_agent(req.query, max_turns=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed_ms = (time.time() - t0) * 1000

    return AgenticResponse(
        query=req.query,
        mode="agentic",
        answer=answer,
        tool_log=[
            ToolCall(
                tool=t["tool"],
                args=t["args"],
                result_preview=t["result_preview"],
            )
            for t in tool_log
        ],
        elapsed_ms=round(elapsed_ms, 1),
    )
