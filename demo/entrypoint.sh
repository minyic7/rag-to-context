#!/bin/bash
set -e

echo "=== RAG Workshop API ==="

# Extract annual report text if not done yet
if [ -f "data/2025-annual-report.pdf" ] && [ ! -f "data/annual_report_text.txt" ]; then
    echo "Extracting annual report PDF → text..."
    pdftotext data/2025-annual-report.pdf data/annual_report_text.txt
    echo "Done."
fi

# Generate embeddings if not done yet (requires OPENAI_API_KEY)
if [ ! -f "data/embeddings.npz" ]; then
    if [ -n "$OPENAI_API_KEY" ]; then
        echo "Generating embeddings (this may take a few minutes on first run)..."
        uv run python embedder.py
        echo "Embeddings ready."
    else
        echo "WARNING: OPENAI_API_KEY not set. Skipping embeddings generation."
        echo "         Only naive mode will be available."
    fi
else
    echo "Embeddings already exist."
fi

echo "Starting API server on :8000 ..."
exec uv run uvicorn server:app --host 0.0.0.0 --port 8000
