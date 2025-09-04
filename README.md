# Maverick RAG - Enterprise Retrieval System

A production-ready RAG (Retrieval-Augmented Generation) application supporting both local development and enterprise Databricks deployment, built according to the project charter specifications.

## Features

- **LLM Providers**: Meta Llama 3.1-8B-Instruct via Together AI (default), OpenAI, Ollama, or Databricks AI/ML Models
- **Vector Store**: FAISS for local development, Databricks Vector Search for enterprise deployment
- **Embeddings**: HuggingFace sentence-transformers (local) or Databricks managed embeddings (enterprise)
- **Data Storage**: Local filesystem or Unity Catalog volumes
- **UI**: Gradio web interface
- **Document Processing**: Automatic text chunking and indexing with Delta Lake integration

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp env.example .env
# Edit .env and add your API keys and configuration
```

### 2. Add Documents

**Local Development:**
Place your text files (`.txt`, `.md`, etc.) in the `data/docs/` directory.

**Databricks Deployment:**
Upload your documents to the Unity Catalog volume specified in `DATABRICKS_VOLUME_PATH`.

### 3. Build Index

**Local Development:**
```bash
python -m src.rag.ingest
```

**Databricks Deployment:**
```bash
export USE_DATABRICKS=true
python -m src.rag.ingest
```

### 4. Launch App

```bash
python app/app.py
```

The Gradio interface will open in your browser at `http://localhost:7860`.

## Configuration

The application uses a centralized configuration management system located in `src/utils/config_manager.py`. This provides:

- **Unified Configuration**: All settings managed through a single `ConfigManager` class
- **Validation**: Automatic validation of configuration with detailed error reporting
- **Type Safety**: Strongly typed configuration objects using dataclasses
- **Environment Integration**: Seamless integration with environment variables

### Configuration Manager Usage

```python
from src.utils import config_manager

# Access different configuration sections
llm_config = config_manager.llm
data_config = config_manager.data
app_config = config_manager.app

# Validate configuration
validation = config_manager.validate_config()
if not validation["valid"]:
    print(f"Configuration errors: {validation['errors']}")

# Get configuration summary
summary = config_manager.get_config_summary()
```

### Environment Variables

**Core Configuration:**
- `LLM_PROVIDER`: Provider to use (`together`, `openai`, `ollama`, `databricks`)
- `LLAMA_MODEL`: Model name (default: `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `TOGETHER_API_KEY`: Your Together AI API key
- `EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `DOCS_DIR`: Documents directory (default: `data/docs`)
- `INDEX_DIR`: Index storage directory (default: `data/index/faiss`)

**Databricks Configuration:**
- `USE_DATABRICKS`: Enable Databricks integration (`true`/`false`)
- `DATABRICKS_WORKSPACE_URL`: Your Databricks workspace URL
- `DATABRICKS_ACCESS_TOKEN`: Your Databricks access token
- `DATABRICKS_CATALOG`: Unity Catalog name (default: `main`)
- `DATABRICKS_SCHEMA`: Schema name (default: `default`)
- `DATABRICKS_VOLUME`: Volume name (default: `source_data`)
- `DATABRICKS_VOLUME_PATH`: Full volume path (e.g., `/Volumes/main/default/source_data`)
- `DATABRICKS_VECTOR_SEARCH_ENDPOINT`: Vector search endpoint name
- `DATABRICKS_MODEL_SERVING_ENDPOINT`: Model serving endpoint name

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

**Databricks AI/ML Models:**
```bash
export LLM_PROVIDER=databricks
export USE_DATABRICKS=true
export DATABRICKS_WORKSPACE_URL=your_workspace.databricks.com
export DATABRICKS_ACCESS_TOKEN=your_token
export DATABRICKS_MODEL_SERVING_ENDPOINT=your_endpoint
```

## Project Structure

```
├── app/
│   └── app.py              # Gradio UI
├── src/
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── llm_provider.py # LLM configuration
│   │   ├── ingest.py       # Document ingestion
│   │   └── chain.py        # RAG chain logic
│   └── utils/
│       ├── __init__.py
│       ├── config_manager.py    # Centralized configuration management
│       └── databricks_config.py # Databricks integration
├── data/
│   ├── docs/               # Input documents (local)
│   └── index/              # FAISS indices (local)
├── tests/
│   └── test_rag.py         # Basic tests
├── env.example             # Environment configuration template
└── requirements.txt
```

## Testing

```bash
python -m pytest tests/
```

## Architecture Alignment

This implementation follows the project charter's four-layer architecture:

- **Data Foundation**: Unity Catalog volumes with Delta Lake tables and Databricks Vector Search
- **Tooling & Integration**: LangChain framework with Databricks SDK integration
- **Intelligence & Reasoning**: RAG chain with configurable LLM providers
- **Foundational LLM**: Meta Llama via Together AI, OpenAI, Ollama, or Databricks AI/ML Models

## Databricks Integration

### Unity Catalog Setup

1. **Create Volume:**
   ```sql
   CREATE VOLUME IF NOT EXISTS main.default.source_data;
   ```

2. **Upload Documents:**
   Upload your documents to the Unity Catalog volume using Databricks UI or CLI.

3. **Create Vector Search Index:**
   Use the Databricks Vector Search UI to create an index on your documents table with managed embeddings.

4. **Deploy Model Serving Endpoint:**
   Deploy your preferred LLM model to a Databricks Model Serving endpoint.

### Environment Configuration

Set the following environment variables for Databricks deployment:

```bash
export USE_DATABRICKS=true
export DATABRICKS_WORKSPACE_URL=your_workspace.databricks.com
export DATABRICKS_ACCESS_TOKEN=your_access_token
export DATABRICKS_CATALOG=main
export DATABRICKS_SCHEMA=default
export DATABRICKS_VOLUME=source_data
export DATABRICKS_VECTOR_SEARCH_ENDPOINT=your_vector_search_endpoint
export DATABRICKS_MODEL_SERVING_ENDPOINT=your_model_serving_endpoint
```

## Next Steps

- Add GraphRAG knowledge graph capabilities with Spark GraphX
- Implement MCP (Model Context Protocol) integration
- Add OpenAI Assistants API for multi-agent orchestration
- Enhanced monitoring and observability with MLflow
