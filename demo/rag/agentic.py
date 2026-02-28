"""
agentic.py — Agentic RAG: Multi-step tool-calling agent.

The agent decides WHICH tool to call and HOW MANY times, based on the query.
Tools available:
  - search_web_pages(query)      → BM25 + dense hybrid over crawled pages
  - lookup_financials(metric)    → structured lookup in financial CSVs
  - follow_links(url)            → fetch a specific page's text
  - read_annual_report(question) → semantic search inside the PDF text

The agent can chain calls: e.g. "What is CBA's dividend yield?" →
  1. lookup_financials("dividend yield")
  2. search_web_pages("CBA dividend policy")
  → synthesise answer from both

This is the most capable approach but also most expensive (multiple LLM calls).
"""

from __future__ import annotations
import json
import os
from pathlib import Path

# ── Tool implementations ──────────────────────────────────────────────────────

def _get_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        env = Path(__file__).parent.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    return OpenAI(api_key=api_key)


def _tool_search_web(query: str) -> str:
    """BM25 + dense hybrid search over crawled CommBank pages."""
    from .naive import retrieve as bm25_retrieve
    try:
        from .advanced import retrieve as dense_retrieve
        dense_results = dense_retrieve(query, top_k=3)
    except RuntimeError:
        dense_results = []

    bm25_results = bm25_retrieve(query, top_k=3)

    seen: set[str] = set()
    combined = []
    for r in dense_results + bm25_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            combined.append(r)

    if not combined:
        return "No results found."

    snippets = []
    for r in combined[:4]:
        snippets.append(f"[{r['title'] or r['url']}]\n{r['text'][:600]}")
    return "\n\n---\n\n".join(snippets)


def _tool_lookup_financials(metric: str) -> str:
    """Search financial CSVs for a specific metric or keyword."""
    data_dir = Path(__file__).parent.parent / "data"
    results = []
    for csv_file in ["key_metrics.csv", "income_statement.csv",
                     "balance_sheet.csv", "cash_flow.csv"]:
        path = data_dir / csv_file
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Find rows mentioning the metric (case-insensitive)
        matching = [
            line for line in text.splitlines()
            if metric.lower() in line.lower()
        ]
        if matching:
            results.append(f"=== {csv_file} ===\n" + "\n".join(matching[:10]))
    return "\n\n".join(results) if results else f"No financial data found for '{metric}'."


def _tool_follow_link(url: str) -> str:
    """Return the text content of a specific crawled page."""
    from .corpus import load_page_texts
    texts = load_page_texts()
    text  = texts.get(url, "")
    if not text:
        return f"Page not found in corpus: {url}"
    return text[:3000]


def _tool_read_annual_report(question: str) -> str:
    """
    Search the annual report PDF text for relevant sections.
    Uses BM25 over 500-char chunks of the PDF text.
    """
    import re, math
    from collections import Counter

    pdf_text_path = Path(__file__).parent.parent / "data" / "annual_report_text.txt"
    if not pdf_text_path.exists():
        return (
            "Annual report text not extracted yet. "
            "Run: pdftotext data/2025-annual-report.pdf data/annual_report_text.txt"
        )

    text = pdf_text_path.read_text(encoding="utf-8", errors="ignore")
    # Split into ~500-char chunks
    chunk_size = 500
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size // 6):  # ~6 chars/word avg
        chunk = " ".join(words[i:i + 80])
        if chunk.strip():
            chunks.append(chunk)

    # Simple BM25
    def tokenise(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    q_toks = tokenise(question)
    scored = []
    for chunk in chunks:
        c_toks = tokenise(chunk)
        tf = Counter(c_toks)
        score = sum(tf.get(t, 0) for t in q_toks)
        scored.append((score, chunk))

    scored.sort(key=lambda x: -x[0])
    top = [c for _, c in scored[:3] if _]
    return "\n\n---\n\n".join(top) if top else "No relevant sections found."


# ── Tool registry for the agent ───────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web_pages",
            "description": "Search CommBank website pages for information about products, services, rates, and policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_financials",
            "description": "Look up CBA financial metrics from annual reports: revenue, profit, EPS, dividend yield, P/E ratio, balance sheet items, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "description": "Financial metric or keyword to search for"}
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "follow_link",
            "description": "Retrieve the full text of a specific CommBank webpage by its URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL of the page to retrieve"}
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_annual_report",
            "description": "Search the CBA 2025 Annual Report PDF for specific information about strategy, risk, governance, or detailed financials.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "What to look for in the annual report"}
                },
                "required": ["question"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "search_web_pages": lambda args: _tool_search_web(args["query"]),
    "lookup_financials": lambda args: _tool_lookup_financials(args["metric"]),
    "follow_link":      lambda args: _tool_follow_link(args["url"]),
    "read_annual_report": lambda args: _tool_read_annual_report(args["question"]),
}


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(
    query: str,
    max_turns: int = 5,
    on_tool_call=None,   # optional callback(tool_name, args) for UI
) -> tuple[str, list[dict]]:
    """
    Run the agentic RAG loop.

    Returns:
      (answer, tool_call_log)
      tool_call_log: list of {tool, args, result_preview}
    """
    client = _get_client()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable assistant with access to CommBank's "
                "website content, financial data, and the 2025 Annual Report. "
                "Use the tools to gather relevant information before answering. "
                "Cite your sources. Be concise and accurate."
            ),
        },
        {"role": "user", "content": query},
    ]

    tool_log = []

    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            # Agent is done
            return msg.content or "", tool_log

        # Execute tool calls
        messages.append(msg)   # append assistant message with tool_calls
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if on_tool_call:
                on_tool_call(fn_name, fn_args)

            handler = TOOL_DISPATCH.get(fn_name)
            result  = handler(fn_args) if handler else f"Unknown tool: {fn_name}"

            tool_log.append({
                "tool":           fn_name,
                "args":           fn_args,
                "result_preview": result[:300],
            })

            messages.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "content":      result,
            })

    # Fallback: ask for a final answer without tools
    messages.append({
        "role": "user",
        "content": "Please provide your best answer based on the information gathered."
    })
    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return final.choices[0].message.content or "", tool_log
