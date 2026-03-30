# LangChain RAG Pipeline - Context & Architecture Guide

## Quick Project Summary

**Purpose:** Local Retrieval-Augmented Generation (RAG) system using Ollama, LangChain, and Chroma.
- PDF ingestion and embedding  
- Vector search via Chroma DB  
- LLM-powered Q&A with source citations  
- CLI-based interaction

**Repository:** `pardeepc4u/langchain-rag-pipeline`  
**Current Branch:** `feature/poetry`  
**Default Branch:** `main`  
**Python Version:** 3.12  
**Status:** Working (all tests pass)

---

## Project Structure

```
langchain-rag-pipeline/
├── pyproject.toml           # Poetry config (CRITICAL: see Dependency Issues)
├── poetry.lock              # Lock file (auto-generated)
├── README.md                # User documentation
├── context.md               # THIS FILE - AI bootstrap guide
│
├── src/                     # Main package (requires __init__.py in all subfolders)
│   ├── __init__.py
│   ├── main.py              # CLI entry point (use: python -m src.main)
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py      # Pydantic settings model (ENV vars, paths, LLM config)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── embeddings.py    # OllamaEmbeddings initialization
│   │   └── llm.py           # LLM initialization (Ollama)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py        # PDF loading + text splitting
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   └── vectorstore.py   # Chroma DB connector
│   │
│   └── rag/
│       ├── __init__.py
│       └── chain.py         # QA chain (LCEL RunnableParallel pattern)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures (mocks for embeddings, chain)
│   └── test_rag_pipeline.py # Test: document splitting
│
├── data/
│   └── documents/           # PDF input directory
│
└── chroma_db/               # Vector store persistence (auto-created)
```

---

## Architecture & Data Flow

```
User Input → CLI (click) → main.py
                ↓
        ┌───────────────┬─────────────┐
        ↓               ↓             ↓
   [ingest]        [query]       [help]
        ↓               ↓
  load_documents()  get_qa_chain()
        ↓               ↓
  split_documents() chain.invoke()
        ↓               ↓
  Chroma.add_docs() LLM response
        ↓               ↓
   Vector DB ←────→ Retriever
                    + Prompt
                    + Citation
```

### Key Components

#### 1. `src/config/settings.py`
**Pydantic model** for environment configuration:
- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_EMBED_MODEL` (default: `nomic-embed-text`)
- `OLLAMA_LLM_MODEL` (default: `llama3:8b`)
- `CHROMA_PERSIST_DIR` (default: `./chroma_db`)
- `TOP_K_RESULTS` (default: `4`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`

#### 2. `src/core/llm.py` & `src/core/embeddings.py`
Lazy singletons for LLM and embedding model initialization via Ollama.

#### 3. `src/ingestion/loader.py`
- `load_documents()` → DirectoryLoader + PyPDFLoader  
- `split_documents()` → RecursiveCharacterTextSplitter

#### 4. `src/rag/chain.py` (CRITICAL - Modern LCEL pattern)
**Modern LangChain approach** (NOT deprecated `RetrievalQA`):
- Uses `RunnableParallel` to build chain branches  
- Retriever → prompt → LLM → StrOutputParser  
- Returns: `{"result": str, "source_documents": List[Document]}`
- **Import note:** Uses `from src.core.llm` (not relative `from core`)

#### 5. `src/main.py` (CLI entry point)
Click-based CLI with three commands:
- `ingest` — Load PDFs and build vector DB  
- `query <question>` — Ask a question, get LLM response + sources  
- `--help` — Show command list

---

## Critical Known Issues & Solutions

### Issue 1: NumPy 2.x + Chromadb Incompatibility

**Problem:** `chromadb` v0.4.x uses `np.float_` which was removed in NumPy 2.0+.

**Error:**
```
AttributeError: `np.float_` was removed in the NumPy 2.0 release.
```

**Solution:** Pin compatible versions in `pyproject.toml`:
```toml
langchain-community = "0.3.31"  # (not 0.4.1; 0.4.x forces numpy>=2.1)
chromadb = "^0.4.24"
numpy = ">=1.26.2,<2.0"
```

### Issue 2: Import Path Mismatch

**Problem:** Mixed relative vs absolute imports cause `ModuleNotFoundError`.

**Examples of WRONG patterns:**
```python
from core.llm import get_llm           # ❌ NO - missing src prefix
from config.settings import settings   # ❌ NO - missing src prefix
from .llm import get_llm              # ❌ NO - relative imports + package mode
```

**CORRECT pattern (always use absolute from package root):**
```python
from src.core.llm import get_llm
from src.config.settings import settings
from src.storage.vectorstore import get_vectorstore
```

**Why:** When running `python -m src.main`, Python adds repo root to `sys.path`, making `src.*` the canonical import path.

### Issue 3: Missing `__init__.py` Files

**Problem:** Python doesn't treat directories as packages without `__init__.py`.

**Solution:** All package directories already have `__init__.py` with docstrings:
```python
# src/core/__init__.py
"""core package for embeddings and llm logic."""
```

**Directories with `__init__.py`:**
- `src/`  
- `src/ingestion/`  
- `src/core/`  
- `src/config/`  
- `src/rag/`  
- `src/storage/`  

### Issue 4: `onnxruntime` Wheel Compatibility

**Problem:** Poetry fails with "ABI tag mismatch" for `onnxruntime` on macOS x86_64.

**Solution:** Install manually after Poetry environment setup:
```bash
poetry run pip install onnxruntime==1.23.2
```

**Why:** Pip detects the environment correctly; Poetry's resolver sometimes skips compatible wheels.

---

## Running the Project

### 1. Setup (First Time)

```bash
# Install poetry (if not present)
curl -sSL https://install.python-poetry.org | python3 -

# Navigate to repo
cd /Users/kiyaanyadav/workspace/github_workspace/langchain-rag-pipeline

# Install dependencies
poetry install

# If onnxruntime fails, manual install:
poetry run pip install onnxruntime==1.23.2
```

### 2. Start Ollama Server

```bash
ollama serve
# or if already running, skip this step
```

### 3. Ingest PDFs

```bash
poetry run python -m src.main ingest
# Outputs: ✅ Ingestion completed successfully (or error if no PDFs found)
```

### 4. Query the RAG System

```bash
poetry run python -m src.main query "What is LangChain?"
# Returns: ANSWER + SOURCES (with page numbers and context snippets)
```

### 5. Run Tests

```bash
poetry run pytest tests/ -q
# Expected: 1 passed
```

---

## Running Without Poetry (Direct Python)

If Poetry setup fails irreparably:

```bash
# Use venv directly
cd /Users/kiyaanyadav/workspace/github_workspace/langchain-rag-pipeline
source /Users/kiyaanyadav/.local/share/virtualenvs/langchain-rag-pipeline-Cx9aJtyA/bin/activate

# Run CLI
python -m src.main --help
python -m src.main ingest
python -m src.main query "Your question"

# Run tests
pytest tests/ -q
```

---

## Development Workflow

### Adding a New Feature

1. **Create module in `src/`** → ensure `__init__.py` exists  
2. **Use absolute imports:** `from src.<package>.<module> import <func>`  
3. **Update tests** in `tests/test_*.py`  
4. **Run:** `poetry run pytest tests/ -q`

### Testing

- **Framework:** pytest  
- **Fixtures:** `conftest.py` has mocks for embeddings + chain  
- **Run all:** `poetry run pytest tests/ -q`  
- **Run one file:** `poetry run pytest tests/test_rag_pipeline.py -v`

### Code Quality

- **Formatter:** Black (configured in `pyproject.toml`)  
- **Linter:** MyPy (static type checking)  
- **Pre-commit hooks:** Run before committing

---

## Key Dependency Versions

| Package | Version | Reason |
|---------|---------|--------|
| `langchain-community` | `0.3.31` | Avoids numpy 2.1 hard requirement |
| `chromadb` | `^0.4.24` | Stable, works with numpy 1.26 |
| `numpy` | `>=1.26.2,<2.0` | Chromadb compatibility |
| `langchain` | `^1.2.13` | Core LLM framework |
| `langchain-core` | `^1.2.23` | LCEL runnables (modern patterns) |
| `ollama` | `^0.1.0` | Ollama client  |
| `pypdf` | `^6.9.2` | PDF parsing |
| `pydantic` | `^2.8.1` | Config validation |
| `click` | `^8.1.7` | CLI framework |

---

## Environment Variables (.env)

Create `.env` in project root:

```ini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3:8b
CHROMA_PERSIST_DIR=./chroma_db
TOP_K_RESULTS=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

Loaded by `pydantic-settings` in `src/config/settings.py`.

---

## Common AI Model Tasks

### Task: Add new CLI command
1. Open `src/main.py`  
2. Add `@cli.command()` function  
3. Use click decorators for args/options  
4. Import helpers from `src.*` packages  
5. Test: `poetry run python -m src.main <new-command> --help`

### Task: Modify RAG chain behavior
1. Edit `src/rag/chain.py` (`get_qa_chain()`)  
2. Adjust prompt, retriever kwargs, or chain topology  
3. Test: `poetry run pytest tests/ -q`

### Task: Change LLM or embedding model
1. Edit `.env`: update `OLLAMA_LLM_MODEL` or `OLLAMA_EMBED_MODEL`  
2. Or modify `src/config/settings.py` defaults  
3. No code changes needed

### Task: Handle import errors
1. Check file lives in `src/<package>/` (not random dir)  
2. Verify `__init__.py` exists in that package  
3. Use absolute imports: `from src.<package>.<module> import <func>`  
4. Run: `poetry run python -m src.main --help` to test

---

## Troubleshooting Checklist

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: No module named 'src'` | Script mode vs module mode | Use `python -m src.main`, not `python src/main.py` |
| `AttributeError: np.float_` | NumPy 2.x + chromadb 0.4 conflict | Downgrade NumPy to 1.26.2 (see pyproject.toml) |
| `ImportError: cannot import name 'X' from src.Y` | Wrong import path (missing `src.` prefix) | Update to absolute import, e.g. `from src.Y import X` |
| `FileNotFoundError: Vector DB not found` | Ingestion not run first | Run `poetry run python -m src.main ingest` |
| `poetry lock` hangs | Dependency conflict (numpy vs langchain-community) | Use langchain-community 0.3.31 (not 0.4.1) |

---

## Git Workflow

**Current Branch:** `feature/poetry`  
**Default Branch:** `main`  

```bash
# Create feature branch
git checkout -b feature/<name>

# Commit changes
git add .
git commit -m "feat: <description>"

# Push and create PR
git push -u origin feature/<name>
gh pr create --base main --head feature/<name> --title "feature/<name>" --body "Description"
```

---

## Files You'll Likely Edit

1. **`src/config/settings.py`** — Change default LLM/embedding models or paths  
2. **`src/rag/chain.py`** — Modify RAG chain logic or prompts  
3. **`src/main.py`** — Add/modify CLI commands  
4. **`pyproject.toml`** — Update dependencies (carefully, watch for conflicts)  
5. **`README.md`** — Update user-facing docs  

---

## Notes for Future AI Models

- **Always use absolute imports** from `src.*` root when running `python -m src.main`
- **Check NumPy + ChromaDB version compatibility** before updating dependencies
- **Run tests after any code change:** `poetry run pytest tests/ -q`
- **The chain pattern in `src/rag/chain.py` is modern LCEL**, not legacy `RetrievalQA`
- **All package dirs must have `__init__.py`** (already in place)
- **Ollama must be running** for ingestion and queries to work
- **If imports fail, first check the import path** (missing `src.` prefix is the most common issue)

---

## Last Updated
- **Date:** 2026-03-30  
- **Status:** All tests passing, CLI working, dependencies stable  
- **Known Issues:** None blocking functionality
