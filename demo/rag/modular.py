"""
modular.py — Modular RAG: Dense retrieval + LLM reranking.

Approach:
  1. Dense retrieval (same as advanced.py) → candidate pool (top 20)
  2. LLM reranker scores each candidate 0-10 for relevance to query
  3. Filter low-score candidates (< threshold)
  4. Return top-k reranked chunks

Modular = swap any stage independently (retriever / reranker / generator).
Slower (extra LLM call) but more precise — catches false positives from embeddings.
"""

from __future__ import annotations
import json
import os
from pathlib import Path

from .advanced import retrieve as dense_retrieve, _get_client

RERANK_THRESHOLD = 5   # drop candidates scoring below this


def _rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Ask LLM to score each candidate 0-10 for relevance."""
    client = _get_client()

    # Build a compact scoring prompt
    items = "\n".join(
        f"[{i}] {c['title'] or c['url']}\n{c['text'][:300]}..."
        for i, c in enumerate(candidates)
    )
    prompt = f"""You are a relevance judge. For each document snippet below, rate how relevant it is to answering the query. Output ONLY a JSON array of integers (0-10), one per document, in order.

Query: {query}

Documents:
{items}

Output format: [score0, score1, score2, ...]"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200,
    )
    text = resp.choices[0].message.content.strip()

    # Parse JSON array robustly
    try:
        start = text.index("[")
        end   = text.rindex("]") + 1
        scores = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        # Fall back: return candidates as-is if parsing fails
        return candidates

    # Attach rerank scores
    reranked = []
    for i, candidate in enumerate(candidates):
        rs = scores[i] if i < len(scores) else 0
        reranked.append({**candidate, "rerank_score": rs, "method": "modular"})

    # Filter and sort
    reranked = [c for c in reranked if c["rerank_score"] >= RERANK_THRESHOLD]
    reranked.sort(key=lambda x: -x["rerank_score"])
    return reranked


def retrieve(query: str, top_k: int = 5, candidate_pool: int = 20) -> list[dict]:
    """Dense retrieve → LLM rerank → return top_k."""
    candidates = dense_retrieve(query, top_k=candidate_pool)
    reranked   = _rerank(query, candidates)
    return reranked[:top_k]
