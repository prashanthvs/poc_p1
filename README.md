# Maverick RAG - Llama-Powered Retrieval System

A production-ready RAG (Retrieval-Augmented Generation) application using Meta Llama 3.1 via Together AI, built according to the project charter specifications.

## Features

- **LLM Provider**: Meta Llama 3.1-8B-Instruct via Together AI (default)
- **Vector Store**: FAISS for fast similarity search
- **Embeddings**: HuggingFace sentence-transformers
- **UI**: Gradio web interface
- **Document Processing**: Automatic text chunking and indexing

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY
```

### 2. Add Documents

Place your text files (`.txt`, `.md`, etc.) in the `data/docs/` directory.

### 3. Build Index

```bash
python -m src.rag.ingest
```

### 4. Launch App

```bash
python app/app.py
```

The Gradio interface will open in your browser at `http://localhost:7860`.

## Configuration

### Environment Variables

- `LLM_PROVIDER`: Provider to use (`together`, `openai`, `ollama`)
- `LLAMA_MODEL`: Model name (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `TOGETHER_API_KEY`: Your Together AI API key
- `EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `DOCS_DIR`: Documents directory (default: `data/docs`)
- `INDEX_DIR`: Index storage directory (default: `data/index/faiss`)

### Alternative LLM Providers

**OpenAI:**
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_key_here
```

**Local Ollama:**
```bash
export LLM_PROVIDER=ollama
export LLAMA_MODEL=llama3.1
# Ensure Ollama is running: ollama serve
```

## Project Structure

```
├── app/
│   └── app.py              # Gradio UI
├── src/rag/
│   ├── __init__.py
│   ├── llm_provider.py     # LLM configuration
│   ├── ingest.py           # Document ingestion
│   └── chain.py            # RAG chain logic
├── data/
│   ├── docs/               # Input documents
│   └── index/              # FAISS indices
├── tests/
│   └── test_rag.py         # Basic tests
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/
```

## Architecture Alignment

This implementation follows the project charter's four-layer architecture:

- **Data Foundation**: FAISS vector store (simplified from GraphRAG)
- **Tooling & Integration**: LangChain framework
- **Intelligence & Reasoning**: RAG chain with Llama 3.1
- **Foundational LLM**: Meta Llama hosted via Together AI (North America)

## Next Steps

- Integrate with Databricks and Delta Lake
- Add GraphRAG knowledge graph capabilities
- Implement MCP (Model Context Protocol) integration
- Add OpenAI Assistants API for multi-agent orchestration
