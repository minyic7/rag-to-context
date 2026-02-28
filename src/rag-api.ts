/**
 * rag-api.ts — Client for the RAG Workshop Docker backend.
 */

const RAG_API_BASE = "http://localhost:8000";

export interface ChunkResult {
  url: string;
  title: string;
  text: string;
  score: number;
  method: string;
  source?: string;
  rerank_score?: number;
  via?: string;
}

export interface RetrieveResponse {
  query: string;
  mode: string;
  chunks: ChunkResult[];
  elapsed_ms: number;
}

export interface ToolCall {
  tool: string;
  args: Record<string, string>;
  result_preview: string;
}

export interface AgenticResponse {
  query: string;
  mode: string;
  answer: string;
  tool_log: ToolCall[];
  elapsed_ms: number;
}

export interface HealthResponse {
  status: string;
  available_modes: string[];
  embeddings_ready: boolean;
  annual_report_ready: boolean;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${RAG_API_BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function queryRAG(
  mode: string,
  query: string,
  topK = 5,
): Promise<RetrieveResponse | AgenticResponse> {
  const res = await fetch(`${RAG_API_BASE}/api/${mode}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `API error: ${res.status}`);
  }
  return res.json();
}
