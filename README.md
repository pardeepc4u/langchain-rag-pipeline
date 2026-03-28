# LangChain RAG Pipeline

A Retrieval-Augmented Generation (RAG) system using Ollama, LangChain, and Chroma for local embedding + vector search.

## Description

This repository showcases a RAG pipeline:

1. PDF ingestion from `data/` into Chroma vector store.
2. Ollama embeddings + LLM usage via `langchain-community`.
3. Query answering with vector retrieval and context formatting.
4. Modular config via `src/config/config.py`.

## Tech Stack

- Python 3.8+
- LangChain (1.x) & `langchain-core`, `langchain-community`
- Ollama (`ollama` package)
- ChromaDB (`chromadb`)
- PDF parser (`pypdf`)
- dotenv config (`python-dotenv`)

## Repository structure

- `data/`
  - `*.pdf` source docs (e.g. `RoofInvoice505.pdf`)
- `src/config/config.py`
  - `Settings` class and `settings` instance
- `src/ingestion.py` (legacy ingestion path)
- `src/ollama/ingestion.py` (Ollama-specific ingestion class)
- `src/rag_system.py` (RAG runtime and query loop)

## Configuration

1. Copy `.env.example` or create `.env` in project root.
2. Set Ollama + Chroma:

```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3:8b
CHROMA_PERSIST_DIR=./chroma_db
```

3. Optional:

- `CHUNK_SIZE` and `CHUNK_OVERLAP` are set in `src/config/config.py`.

## Setup

```bash
cd /Users/kiyaanyadav/workspace/github_workspace/langchain-rag-pipeline
python3 -m pip install -r requirements.txt
```

## Start Ollama

```bash
ollama server start
```

(If Ollama is already running, skip this.)

## Ingest PDFs & Build Vector Store

```bash
cd src
python3 ingestion.py  # or python3 ollama/ingestion.py
```

`ingestion.py` uses:
- `DirectoryLoader` with `glob="**/*.pdf"`
- `PyPDFLoader`
- `RecursiveCharacterTextSplitter` with config chunk/overlap
- `Chroma` with persistence at `settings.CHROMA_PERSIST_DIR`

## Run RAG Query Loop

```bash
cd src
python3 rag_system.py
```

Then ask questions interactively.

## Common issues

- `Import "langchain.chains" could not be resolved`: install dependencies and use correct module paths (`langchain_core` or `langchain_community`).
- `settings` unknown symbol: use `from src.config.config import settings` or relative module fixes.
- `file not found` on `data/`: ensure the path exists and contains at least one PDF.

## GitHub CLI PR flow

```bash
git checkout -b feature/ollama_init
git add .
git commit -m "Update RAG pipeline config with Ollama + Chroma"
git push -u origin feature/ollama_init
gh pr create --base main --head feature/ollama_init --title "feature/ollama_init" --body "Add Ollama RAG pipeline" --draft
```

## Notes

- Keep `langchain` and `langchain-core` versions aligned to avoid import mismatches.
- Use `git check-ignore -v <file>` to validate `.gitignore` rules.
- This repo is designed for local RAG experimentation, not production LLM serving.
