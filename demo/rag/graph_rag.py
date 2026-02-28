"""
graph_rag.py — GraphRAG: Seed retrieval + link-graph expansion.

Approach:
  1. Dense retrieval → seed nodes (top 5)
  2. Expand via edges in graph.json → 1-hop neighbours
  3. Score neighbours by edge proximity + embedding similarity
  4. Return merged, deduplicated context

GraphRAG captures related pages (e.g. "home loans" seed → expands to
"fixed rate", "offset account", "loan calculator" neighbours).
Especially powerful for multi-hop questions spanning several pages.
"""

from __future__ import annotations
from collections import defaultdict

from .corpus import load_edges, load_page_texts, load_nodes
from .advanced import retrieve as dense_retrieve, _cosine_sim, _embed_query, load_embeddings

NEIGHBOUR_SCORE_BOOST = 0.1   # score bonus per shared edge with a seed


def _build_adjacency(edges: list[dict]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e["from"]].add(e["to"])
        adj[e["to"]].add(e["from"])   # treat as undirected
    return adj


def retrieve(query: str, top_k: int = 5, seed_k: int = 5) -> list[dict]:
    """Seed dense retrieval then expand via graph edges."""
    seeds    = dense_retrieve(query, top_k=seed_k)
    seed_urls = {s["url"] for s in seeds}

    edges    = load_edges()
    adj      = _build_adjacency(edges)
    page_texts  = load_page_texts()
    nodes_by_url = {n["url"]: n for n in load_nodes()}

    # Gather all neighbour URLs (1-hop from any seed)
    neighbour_counts: dict[str, int] = defaultdict(int)
    for s in seeds:
        for nb_url in adj.get(s["url"], set()):
            if nb_url not in seed_urls:
                neighbour_counts[nb_url] += 1   # count how many seeds link here

    # Embed query once for scoring neighbours
    emb_result = load_embeddings()
    if emb_result is not None:
        matrix, emb_urls = emb_result
        url_to_idx = {u: i for i, u in enumerate(emb_urls)}
        q_vec = _embed_query(query)
        sims  = _cosine_sim(matrix, q_vec)
    else:
        url_to_idx = {}
        sims = None

    # Score neighbours
    neighbour_results = []
    for nb_url, edge_count in neighbour_counts.items():
        text = page_texts.get(nb_url, "")
        if not text:
            continue
        node = nodes_by_url.get(nb_url, {})
        emb_score = 0.0
        if sims is not None and nb_url in url_to_idx:
            emb_score = float(sims[url_to_idx[nb_url]])
        graph_score = emb_score + NEIGHBOUR_SCORE_BOOST * edge_count
        neighbour_results.append({
            "url":    nb_url,
            "title":  node.get("title", ""),
            "source": "web",
            "text":   text[:4000],
            "score":  round(graph_score, 4),
            "method": "graph",
            "via":    "neighbour",
        })

    # Mark seeds
    for s in seeds:
        s["method"] = "graph"
        s["via"]    = "seed"

    # Merge seeds + neighbours, deduplicate, sort by score
    all_results = list(seeds) + neighbour_results
    seen: set[str] = set()
    merged = []
    for r in sorted(all_results, key=lambda x: -x["score"]):
        if r["url"] not in seen:
            seen.add(r["url"])
            merged.append(r)

    return merged[:top_k]
