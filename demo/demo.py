"""
demo.py — Interactive RAG evolution demo (CommBank knowledge base).

Shows all 5 RAG approaches answering the same question:
  1. Naive RAG      — BM25 keyword matching
  2. Advanced RAG   — Dense embeddings (text-embedding-3-large)
  3. Modular RAG    — Dense + LLM reranker
  4. GraphRAG       — Seed + graph expansion
  5. Agentic RAG    — Multi-step tool-calling agent

Usage:
  uv run python demo.py
  uv run python demo.py --question "What is CBA's home loan interest rate?"
  uv run python demo.py --mode naive          # run just one mode
  uv run python demo.py --mode advanced
  uv run python demo.py --mode modular
  uv run python demo.py --mode graph
  uv run python demo.py --mode agentic
"""

import argparse
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

# ── Load .env ─────────────────────────────────────────────────────────────────
_env = Path(__file__).parent / ".env"
if _env.exists() and not os.getenv("OPENAI_API_KEY"):
    for line in _env.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip()
            break

from openai import OpenAI

console = Console()

# ── LLM generation (shared) ───────────────────────────────────────────────────

def generate_answer(query: str, chunks: list[dict], mode_name: str) -> str:
    """Call GPT-4o with retrieved context and return the answer."""
    client = OpenAI()
    context = "\n\n---\n\n".join(
        f"Source: {c.get('title') or c.get('url', '')}\n{c['text'][:1200]}"
        for c in chunks
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant for Commonwealth Bank of Australia. "
                "Answer the question using ONLY the provided context. "
                "If the context is insufficient, say so. Be concise (2-4 sentences)."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}",
        },
    ]
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=400,
    )
    return resp.choices[0].message.content or ""


# ── Mode runners ──────────────────────────────────────────────────────────────

def run_naive(query: str) -> None:
    from rag.naive import retrieve

    console.print(Rule("[bold cyan]1 · Naive RAG[/bold cyan] — BM25 keyword matching"))

    t0 = time.time()
    chunks = retrieve(query, top_k=5)
    retrieve_ms = (time.time() - t0) * 1000

    _print_chunks(chunks, retrieve_ms)

    t0 = time.time()
    answer = generate_answer(query, chunks, "naive")
    gen_ms = (time.time() - t0) * 1000

    _print_answer(answer, gen_ms)


def run_advanced(query: str) -> None:
    from rag.advanced import retrieve

    console.print(Rule("[bold green]2 · Advanced RAG[/bold green] — Dense embeddings"))

    t0 = time.time()
    chunks = retrieve(query, top_k=5)
    retrieve_ms = (time.time() - t0) * 1000

    _print_chunks(chunks, retrieve_ms)

    t0 = time.time()
    answer = generate_answer(query, chunks, "advanced")
    gen_ms = (time.time() - t0) * 1000

    _print_answer(answer, gen_ms)


def run_modular(query: str) -> None:
    from rag.modular import retrieve

    console.print(Rule("[bold yellow]3 · Modular RAG[/bold yellow] — Dense + LLM reranker"))

    t0 = time.time()
    chunks = retrieve(query, top_k=5)
    retrieve_ms = (time.time() - t0) * 1000

    _print_chunks(chunks, retrieve_ms)

    t0 = time.time()
    answer = generate_answer(query, chunks, "modular")
    gen_ms = (time.time() - t0) * 1000

    _print_answer(answer, gen_ms)


def run_graph(query: str) -> None:
    from rag.graph_rag import retrieve

    console.print(Rule("[bold magenta]4 · GraphRAG[/bold magenta] — Graph-expanded retrieval"))

    t0 = time.time()
    chunks = retrieve(query, top_k=5)
    retrieve_ms = (time.time() - t0) * 1000

    _print_chunks(chunks, retrieve_ms)

    t0 = time.time()
    answer = generate_answer(query, chunks, "graph")
    gen_ms = (time.time() - t0) * 1000

    _print_answer(answer, gen_ms)


def run_agentic(query: str) -> None:
    from rag.agentic import run_agent

    console.print(Rule("[bold red]5 · Agentic RAG[/bold red] — Multi-step tool-calling agent"))

    tool_calls_display = []

    def on_tool_call(tool_name: str, args: dict):
        arg_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
        console.print(f"  [dim]→ calling[/dim] [bold]{tool_name}[/bold]({arg_str})")
        tool_calls_display.append((tool_name, args))

    t0 = time.time()
    answer, tool_log = run_agent(query, on_tool_call=on_tool_call)
    total_ms = (time.time() - t0) * 1000

    if not tool_log:
        console.print("  [dim](no tools called)[/dim]")

    console.print()
    console.print(Panel(
        Text(answer, style="white"),
        title=f"[bold red]Answer[/bold red]  [dim]{total_ms:.0f}ms[/dim]",
        border_style="red",
        padding=(1, 2),
    ))


# ── Display helpers ───────────────────────────────────────────────────────────

def _print_chunks(chunks: list[dict], retrieve_ms: float) -> None:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=7)
    table.add_column("Title / URL", width=40)
    table.add_column("Preview", width=55)

    for i, c in enumerate(chunks, 1):
        score = c.get("rerank_score", c.get("score", 0))
        title = (c.get("title") or c.get("url", ""))[:38]
        preview = c["text"].replace("\n", " ")[:120] + "…"
        table.add_row(str(i), f"{score:.2f}", title, preview)

    console.print(f"\n  [dim]Retrieved {len(chunks)} chunks in {retrieve_ms:.0f}ms[/dim]")
    console.print(table)


def _print_answer(answer: str, gen_ms: float) -> None:
    console.print(Panel(
        Text(answer, style="white"),
        title=f"[bold]Answer[/bold]  [dim]{gen_ms:.0f}ms[/dim]",
        border_style="dim",
        padding=(1, 2),
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

DEFAULT_QUESTIONS = [
    "What are CBA's home loan interest rates?",
    "How do I open a CommBank savings account?",
    "What is CBA's net profit for FY2024?",
    "What is CBA's dividend yield and EPS?",
    "How does CommBank protect against fraud and scams?",
]

MODES = {
    "naive":    run_naive,
    "advanced": run_advanced,
    "modular":  run_modular,
    "graph":    run_graph,
    "agentic":  run_agentic,
}


def main():
    parser = argparse.ArgumentParser(description="RAG evolution demo — CommBank")
    parser.add_argument("--question", "-q", default=None, help="Question to ask")
    parser.add_argument(
        "--mode", "-m",
        choices=list(MODES.keys()) + ["all"],
        default="all",
        help="Which RAG mode to run (default: all)"
    )
    args = parser.parse_args()

    # Verify API key
    if not os.getenv("OPENAI_API_KEY"):
        console.print("[red]OPENAI_API_KEY not set. Add it to .env[/red]")
        sys.exit(1)

    # Question selection
    if args.question:
        query = args.question
    else:
        console.print("\n[bold]Select a demo question:[/bold]\n")
        for i, q in enumerate(DEFAULT_QUESTIONS, 1):
            console.print(f"  {i}. {q}")
        console.print()
        choice = console.input("Enter number (or type your own question): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(DEFAULT_QUESTIONS):
            query = DEFAULT_QUESTIONS[int(choice) - 1]
        else:
            query = choice

    console.print()
    console.print(Panel(
        f"[bold white]{query}[/bold white]",
        title="[bold]Question[/bold]",
        border_style="bright_white",
        padding=(0, 2),
    ))
    console.print()

    # Run selected mode(s)
    if args.mode == "all":
        run_order = list(MODES.keys())
    else:
        run_order = [args.mode]

    for mode in run_order:
        MODES[mode](query)
        console.print()


if __name__ == "__main__":
    main()
