# Maverick RAG - Interactive Notebooks

This directory contains Jupyter notebooks for interactive development and testing of the Maverick RAG system.

## Notebooks Overview

### 1. **01_data_preparation.ipynb**
- Document loading and preprocessing
- Text chunking and splitting
- Vector index creation (FAISS and Databricks)
- Unity Catalog volume management
- Delta Lake table creation

### 2. **02_rag_development.ipynb**
- RAG chain testing and development
- Retriever performance evaluation
- LLM provider comparison
- Prompt engineering and optimization
- Vector search experimentation

### 3. **03_app_deployment.ipynb**
- Gradio interface testing
- Configuration validation
- Deployment testing
- UI customization
- Performance monitoring

### 4. **04_evaluation.ipynb**
- RAG system evaluation metrics
- Retrieval accuracy testing
- Response quality assessment
- A/B testing framework
- Performance benchmarking

## Usage

1. **Install Jupyter**: `pip install jupyter notebook`
2. **Start Jupyter**: `jupyter notebook` or `jupyter lab`
3. **Navigate to notebooks folder** and open the desired notebook
4. **Follow the setup instructions** in each notebook

## Prerequisites

- All dependencies from `requirements.txt` installed
- Environment variables configured (see `env.example`)
- Documents in `data/docs/` or Unity Catalog volume configured

## Tips

- Run notebooks in order for best results
- Each notebook is self-contained with setup instructions
- Use the configuration manager for consistent settings
- Save your work frequently when experimenting
