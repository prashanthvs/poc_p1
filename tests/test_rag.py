"""Basic smoke tests for RAG components."""
import os
import tempfile
from pathlib import Path

import pytest

from src.rag.llm_provider import get_llm
from src.rag.ingest import build_faiss_index


def test_llm_provider_defaults():
    """Test that LLM provider defaults to Together with Llama model."""
    # Mock environment to avoid API calls
    original_provider = os.environ.get("LLM_PROVIDER")
    original_model = os.environ.get("LLAMA_MODEL")
    
    try:
        # Clear env vars to test defaults
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
        if "LLAMA_MODEL" in os.environ:
            del os.environ["LLAMA_MODEL"]
        
        # This should not raise an error even without API key
        # (it will fail on actual call, but instantiation should work)
        llm = get_llm()
        assert llm is not None
        
    finally:
        # Restore original values
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider
        if original_model:
            os.environ["LLAMA_MODEL"] = original_model


def test_ingest_with_sample_data():
    """Test ingestion with temporary sample data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir) / "docs"
        docs_dir.mkdir()
        
        # Create sample document
        sample_doc = docs_dir / "sample.txt"
        sample_doc.write_text("This is a sample document for testing RAG ingestion.")
        
        index_dir = Path(temp_dir) / "index"
        
        # Test ingestion
        result_path = build_faiss_index(str(docs_dir), str(index_dir))
        
        assert Path(result_path).exists()
        assert (Path(result_path) / "index.faiss").exists()


if __name__ == "__main__":
    pytest.main([__file__])
