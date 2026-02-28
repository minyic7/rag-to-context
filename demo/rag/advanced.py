"""
advanced.py — Advanced RAG: Dense vector retrieval (OpenAI embeddings).

Approach:
  1. Load pre-computed embeddings.npz (text-embedding-3-large, 3072-dim)
  2. Embed the query with the same model
  3. Cosine similarity → top-k chunks
  4. Call LLM with [context + question]

Semantic: finds "home loan interest rate" even when page says "mortgage rate".
Falls on freshness (stale embeddings) and out-of-corpus knowledge.
"""

from __future__ import annotations
import os
from pathlib import Path

import numpy as np

from .corpus import load_embeddings, load_page_texts, load_nodes, all_chunks

MODEL = "text-embedding-3-large"
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            env = Path(__file__).parent.parent / ".env"
            if env.exists():
                for line in env.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
        _client = OpenAI(api_key=api_key)
    return _client


def _embed_query(query: str) -> np.ndarray:
    client = _get_client()
    resp = client.embeddings.create(model=MODEL, input=[query])
    return np.array(resp.data[0].embedding, dtype=np.float32)


def _cosine_sim(matrix: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """matrix: (N, D), vec: (D,) — returns (N,) similarity scores."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    mat_norm = matrix / np.where(norms == 0, 1, norms)
    vec_norm = vec / (np.linalg.norm(vec) or 1)
    return mat_norm @ vec_norm


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return top_k chunks by cosine similarity to query embedding."""
    result = load_embeddings()
    if result is None:
        raise RuntimeError(
            "embeddings.npz not found. Run: uv run python embedder.py"
        )
    matrix, urls = result

    q_vec = _embed_query(query)
    sims  = _cosine_sim(matrix, q_vec)

    top_indices = np.argsort(-sims)[:top_k]

    page_texts = load_page_texts()
    nodes_by_url = {n["url"]: n for n in load_nodes()}

    results = []
    for idx in top_indices:
        url  = urls[idx]
        node = nodes_by_url.get(url, {})
        text = page_texts.get(url, "")
        results.append({
            "url":    url,
            "title":  node.get("title", ""),
            "source": "web",
            "text":   text[:4000],
            "score":  round(float(sims[idx]), 4),
            "method": "dense",
        })
    return results
