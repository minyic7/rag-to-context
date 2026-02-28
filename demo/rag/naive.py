"""
naive.py — Naive RAG: BM25 keyword retrieval (no embeddings).

Approach:
  1. Tokenise query and every document
  2. Score using TF-IDF-like BM25 formula
  3. Return top-k chunks as context
  4. Call LLM with [context + question]

No semantic understanding — purely lexical matching.
Fails on synonyms, paraphrasing, conceptual queries.
"""

from __future__ import annotations
import math
import re
from collections import Counter

from .corpus import all_chunks

# BM25 hyperparameters
K1 = 1.5
B  = 0.75


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_index(chunks: list[dict]) -> tuple[list[list[str]], float, list[int]]:
    """Returns (token lists, avg_dl, doc_freqs mapping token→count)."""
    tokenised = [_tokenise(c["text"]) for c in chunks]
    avg_dl = sum(len(t) for t in tokenised) / max(len(tokenised), 1)
    # doc frequency per term
    df: Counter = Counter()
    for toks in tokenised:
        df.update(set(toks))
    return tokenised, avg_dl, df


def bm25_score(query_tokens: list[str], doc_tokens: list[str],
               df: Counter, n_docs: int, avg_dl: float) -> float:
    tf = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in tf:
            continue
        idf = math.log((n_docs - df[term] + 0.5) / (df[term] + 0.5) + 1)
        tf_score = (tf[term] * (K1 + 1)) / (tf[term] + K1 * (1 - B + B * dl / avg_dl))
        score += idf * tf_score
    return score


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Return top_k chunks by BM25 score."""
    chunks = all_chunks()
    tokenised, avg_dl, df = _build_index(chunks)
    q_tokens = _tokenise(query)
    n = len(chunks)

    scored = [
        (bm25_score(q_tokens, tokenised[i], df, n, avg_dl), chunks[i])
        for i in range(n)
    ]
    scored.sort(key=lambda x: -x[0])
    results = []
    for score, chunk in scored[:top_k]:
        results.append({**chunk, "score": round(score, 4), "method": "bm25"})
    return results
