"""
embedder.py — Pre-compute OpenAI embeddings for all crawled pages.

Reads every .txt file listed in data/graph.json, calls
text-embedding-3-large in batches, and saves:
  data/embeddings.npz  —  {url: vector} packed into numpy arrays

Usage:
  uv run python embedder.py
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

GRAPH_PATH   = Path("data/graph.json")
PAGES_DIR    = Path("data/pages")
OUT_PATH     = Path("data/embeddings.npz")
MODEL        = "text-embedding-3-large"
BATCH_SIZE   = 64           # tokens per request well under 8192-token limit
MAX_CHARS    = 8000         # truncate page text to ~2000 tokens

console = Console()


def load_chunks() -> list[dict]:
    """Return list of {url, file, text} for every node that has a .txt file."""
    graph = json.loads(GRAPH_PATH.read_text())
    chunks = []
    for node in graph["nodes"]:
        path = Path("data") / node["file"]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        chunks.append({
            "url":  node["url"],
            "file": node["file"],
            "text": text[:MAX_CHARS],
        })
    return chunks


def embed_batches(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in batches; returns list of vectors."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=MODEL, input=batch)
        vectors.extend([r.embedding for r in resp.data])
        # Polite rate-limit pause
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.1)
    return vectors


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Try loading from .env manually
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        console.print("[red]OPENAI_API_KEY not set. Add it to .env or export it.[/red]")
        raise SystemExit(1)

    client = OpenAI(api_key=api_key)

    console.print(f"[bold]Loading pages from {GRAPH_PATH}...[/bold]")
    chunks = load_chunks()
    console.print(f"  {len(chunks)} pages to embed")

    texts = [c["text"] for c in chunks]
    urls  = [c["url"]  for c in chunks]

    console.print(f"\n[bold]Embedding with {MODEL} (batch={BATCH_SIZE})...[/bold]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} batches"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
        task = progress.add_task("Embedding", total=n_batches)

        vectors = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            resp = client.embeddings.create(model=MODEL, input=batch)
            vectors.extend([r.embedding for r in resp.data])
            progress.advance(task)
            if i + BATCH_SIZE < len(texts):
                time.sleep(0.1)

    # Pack into numpy arrays keyed by index; save URL list separately
    matrix = np.array(vectors, dtype=np.float32)  # shape (N, 3072)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        vectors=matrix,
        urls=np.array(urls),
    )
    console.print(f"\n[green]Saved {matrix.shape} embeddings → {OUT_PATH}[/green]")
    total_tokens_est = sum(len(t.split()) * 1.3 for t in texts)
    cost_est = total_tokens_est / 1_000_000 * 0.13
    console.print(f"  Estimated cost: ~${cost_est:.2f} (text-embedding-3-large @ $0.13/1M tokens)")


if __name__ == "__main__":
    main()
