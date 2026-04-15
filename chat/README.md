# Document Chat (RAG)

Minimal React frontend + Python backend for document Q&A through the existing RAG pipeline.

## Run Web Chat

```powershell
cd "C:\Users\grego\Downloads\NLP-Project"
python chat\app.py
```

Then open `http://127.0.0.1:8787`.

## Run CLI Chat

```powershell
python chat\app.py cli
```

## Prerequisites

- Build the index first (`scripts/build_canonical_docs.py`, `scripts/build_chunks.py`, `scripts/build_index.py`)
- Ollama running locally on `localhost:11434`
- Model configured in `rag/config.py` (for example `qwen2.5:7b`)
