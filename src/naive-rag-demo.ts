/**
 * naive-rag-demo.ts — Interactive Naive RAG slide.
 *
 * Renders search UI, calls the backend /api/naive endpoint,
 * and displays BM25 results inside the slide.
 */

import { queryRAG, type RetrieveResponse } from "./rag-api";

const DEFAULT_QUERY = "What is the Digi Home Loan interest rate?";

let initialized = false;

export function initNaiveRagDemo(): void {
  const slide = document.getElementById("slide-naive-rag");
  if (!slide) return;

  const input = slide.querySelector<HTMLInputElement>(".nr-search__input");
  const btn = slide.querySelector<HTMLButtonElement>(".nr-search__btn");
  const results = slide.querySelector<HTMLElement>(".nr-results");
  const status = slide.querySelector<HTMLElement>(".nr-status");
  if (!input || !btn || !results || !status) return;

  const runSearch = async (query: string) => {
    if (!query.trim()) return;
    status.textContent = "Searching...";
    results.innerHTML = "";

    try {
      const data = (await queryRAG("naive", query)) as RetrieveResponse;
      status.textContent = `${data.chunks.length} results · ${data.elapsed_ms.toFixed(0)}ms`;
      renderResults(results, data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      status.textContent = "";
      results.innerHTML = `<div class="nr-error">${msg}<br><span class="nr-error__hint">Is the Docker container running? → docker compose up</span></div>`;
    }
  };

  btn.addEventListener("click", () => runSearch(input.value));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      runSearch(input.value);
    }
  });

  // Stop Reveal.js from capturing keyboard input in the search box
  input.addEventListener("keydown", (e) => e.stopPropagation());

  // Pre-populate with default query on first visit
  if (!initialized) {
    initialized = true;
    input.value = DEFAULT_QUERY;
    runSearch(DEFAULT_QUERY);
  }
}

function renderResults(container: HTMLElement, data: RetrieveResponse): void {
  const html = data.chunks
    .map(
      (c, i) => `
    <div class="nr-result">
      <div class="nr-result__head">
        <span class="nr-result__rank">#${i + 1}</span>
        <span class="nr-result__score">${c.score.toFixed(2)}</span>
        <span class="nr-result__title">${escapeHtml(c.title || c.url)}</span>
      </div>
      <div class="nr-result__text">${escapeHtml(c.text)}</div>
    </div>`,
    )
    .join("");
  container.innerHTML = html;
}

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
