"""
crawl.py — BFS crawler for commbank.com.au
Produces:
  demo/data/graph.json   — nodes + edges graph
  demo/data/pages/       — one .txt per page

Usage:
  python crawler/crawl.py [--root URL] [--depth N] [--out DIR] [--delay SECS]

Defaults:
  root   = https://www.commbank.com.au/
  depth  = 3
  out    = data/
  delay  = 0.5
"""

import argparse
import json
import os
import re
import time
import urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from cleaner import extract_text

# ── Constants ──────────────────────────────────────────────────────────────────

ROOT_URL   = "https://www.commbank.com.au/"
USER_AGENT = "Mozilla/5.0 (educational RAG demo; non-commercial)"

# Only follow links that stay on commbank.com.au and look like content pages
_ALLOWED_HOST   = "www.commbank.com.au"
_SKIP_EXTENSIONS = re.compile(
    r"\.(pdf|jpg|jpeg|png|gif|svg|webp|mp4|zip|doc|docx|xlsx|css|js|xml|json)$",
    re.IGNORECASE,
)
_SKIP_PATH_PREFIX = re.compile(
    r"^/(digital/locate-us|netbank|opensite|search|sitemap|cdn-cgi)",
    re.IGNORECASE,
)


# ── URL helpers ────────────────────────────────────────────────────────────────

def normalise(url: str) -> str:
    """Canonical form: scheme + netloc + path (lowercase), no query/fragment."""
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme.lower(), p.netloc.lower(), path, "", "", ""))


def is_crawlable(url: str) -> bool:
    p = urlparse(url)
    if p.netloc != _ALLOWED_HOST:
        return False
    if _SKIP_EXTENSIONS.search(p.path):
        return False
    if _SKIP_PATH_PREFIX.search(p.path):
        return False
    return True


def url_to_filename(url: str) -> str:
    """Turn a URL into a safe filename (no slashes)."""
    p = urlparse(url)
    slug = (p.path.strip("/").replace("/", "__") or "index")
    slug = re.sub(r"[^\w\-.]", "_", slug)
    return slug[:120] + ".txt"


# ── Robots.txt ─────────────────────────────────────────────────────────────────

def build_robots(root: str) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(urljoin(root, "/robots.txt"))
    try:
        rp.read()
    except Exception:
        pass   # if we can't read robots.txt, proceed permissively
    return rp


# ── Crawler ────────────────────────────────────────────────────────────────────

def crawl(root: str, max_depth: int, out_dir: Path, delay: float) -> dict:
    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    session.headers["Accept-Language"] = "en-AU,en;q=0.9"

    robots = build_robots(root)

    # Graph data
    nodes: dict[str, dict] = {}   # url → node record
    edges: list[dict]       = []  # {from, to}
    edge_set: set[tuple]    = set()

    # BFS queue: (url, depth, parent_url | None)
    queue: deque = deque()
    queue.append((normalise(root), 0, None))
    visited: set[str] = set()

    print(f"Starting crawl: root={root}  max_depth={max_depth}")
    print(f"Output → {out_dir}\n")

    while queue:
        url, depth, parent = queue.popleft()

        if url in visited:
            # Still record the edge even if we've seen this node
            if parent and (parent, url) not in edge_set:
                edges.append({"from": parent, "to": url})
                edge_set.add((parent, url))
            continue

        if depth > max_depth:
            continue

        if not is_crawlable(url):
            continue

        if not robots.can_fetch(USER_AGENT, url):
            print(f"  [robots] skip {url}")
            continue

        visited.add(url)

        # ── Fetch ─────────────────────────────────────────────────────────
        try:
            resp = session.get(url, timeout=10, allow_redirects=True)
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue
        except Exception as e:
            print(f"  [error] {url}: {e}")
            continue

        # ── Clean & save ──────────────────────────────────────────────────
        parsed = extract_text(resp.text, url)
        title  = parsed["title"] or url
        text   = parsed["text"]

        filename = url_to_filename(url)
        filepath = pages_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"URL: {url}\nTitle: {title}\n\n{text}")

        # ── Node record ───────────────────────────────────────────────────
        nodes[url] = {
            "url":   url,
            "title": title,
            "depth": depth,
            "file":  f"pages/{filename}",
            "chars": len(text),
        }

        # ── Edge from parent ──────────────────────────────────────────────
        if parent and (parent, url) not in edge_set:
            edges.append({"from": parent, "to": url})
            edge_set.add((parent, url))

        print(f"  [d{depth}] ({len(nodes):>4} pages)  {title[:60]}")

        # ── Enqueue children ──────────────────────────────────────────────
        if depth < max_depth:
            for href in parsed["links"]:
                child = normalise(urljoin(url, href))
                if child not in visited and is_crawlable(child):
                    queue.append((child, depth + 1, url))

        time.sleep(delay)

    return {
        "meta": {
            "root":        root,
            "max_depth":   max_depth,
            "crawled_at":  datetime.now(timezone.utc).isoformat(),
            "total_pages": len(nodes),
            "total_edges": len(edges),
        },
        "nodes": list(nodes.values()),
        "edges": edges,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BFS crawler → graph.json")
    parser.add_argument("--root",  default=ROOT_URL,   help="Seed URL")
    parser.add_argument("--depth", default=3, type=int, help="Max BFS depth")
    parser.add_argument("--out",   default="data",     help="Output directory")
    parser.add_argument("--delay", default=0.5, type=float, help="Seconds between requests")
    args = parser.parse_args()

    out_dir  = Path(args.out)
    graph    = crawl(args.root, args.depth, out_dir, args.delay)
    out_file = out_dir / "graph.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    m = graph["meta"]
    print(f"\nDone! {m['total_pages']} pages, {m['total_edges']} edges → {out_file}")


if __name__ == "__main__":
    main()
