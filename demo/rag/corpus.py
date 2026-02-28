"""
corpus.py — Shared data loading helpers used by all RAG modules.

Loads:
  - graph.json → nodes/edges
  - pages/*.txt → raw text
  - embeddings.npz → pre-computed vectors (optional, lazy)
  - financial CSVs → structured data snippets
"""

from __future__ import annotations
import json
from pathlib import Path
from functools import lru_cache

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_graph() -> dict:
    return json.loads((DATA_DIR / "graph.json").read_text())


@lru_cache(maxsize=1)
def load_nodes() -> list[dict]:
    return load_graph()["nodes"]


@lru_cache(maxsize=1)
def load_edges() -> list[dict]:
    return load_graph()["edges"]


@lru_cache(maxsize=1)
def load_page_texts() -> dict[str, str]:
    """url → text content"""
    nodes = load_nodes()
    texts = {}
    for node in nodes:
        path = DATA_DIR / node["file"]
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                texts[node["url"]] = text
    return texts


@lru_cache(maxsize=1)
def load_embeddings() -> tuple[np.ndarray, list[str]] | None:
    """Returns (matrix, urls) or None if embeddings.npz not found."""
    emb_path = DATA_DIR / "embeddings.npz"
    if not emb_path.exists():
        return None
    data = np.load(emb_path, allow_pickle=True)
    matrix = data["vectors"].astype(np.float32)
    urls = data["urls"].tolist()
    return matrix, urls


@lru_cache(maxsize=1)
def load_financial_snippets() -> list[dict]:
    """
    Load financial CSVs and return as a list of text snippets for RAG.
    Each snippet is {source, text}.
    """
    snippets = []
    csv_files = {
        "income_statement": "income_statement.csv",
        "balance_sheet":    "balance_sheet.csv",
        "cash_flow":        "cash_flow.csv",
        "key_metrics":      "key_metrics.csv",
    }
    for name, fname in csv_files.items():
        path = DATA_DIR / fname
        if not path.exists():
            continue
        # Read raw CSV as text block — simple but effective for RAG
        text = path.read_text(encoding="utf-8", errors="ignore")
        snippets.append({"source": fname, "url": f"financial://{name}", "text": text})
    return snippets


def all_chunks(include_financials: bool = True) -> list[dict]:
    """
    Returns list of {url, source, text} for every chunk in the corpus.
    Pages are truncated to 4000 chars to keep context windows manageable.
    """
    MAX_CHARS = 4000
    nodes = load_nodes()
    page_texts = load_page_texts()

    chunks = []
    for node in nodes:
        url = node["url"]
        text = page_texts.get(url, "")
        if text:
            chunks.append({
                "url":    url,
                "title":  node.get("title", ""),
                "source": "web",
                "text":   text[:MAX_CHARS],
            })

    if include_financials:
        for s in load_financial_snippets():
            chunks.append({
                "url":    s["url"],
                "title":  s["source"],
                "source": "financial",
                "text":   s["text"][:MAX_CHARS],
            })

    return chunks
