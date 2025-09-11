# src/rag/ingest.py

import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .graph import build_and_save_graph
from utils.config import config_manager as databricks_config

def load_documents(input_dir: str) -> List:
    # (No changes to this function)
    directory = Path(input_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    loader = DirectoryLoader(
        input_dir,
        glob="**/*",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()

def build_faiss_index(docs_dir: str, index_dir: str) -> str:
    # (No changes to this function)
    documents = load_documents(docs_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=200, add_start_index=True
    )
    splits = splitter.split_documents(documents)
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.from_documents(splits, embeddings)
    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(out_dir))
    return str(out_dir)

def build_databricks_artifacts(docs_dir: str):
    """
    Builds all necessary Databricks artifacts:
    1. A Delta table with source document chunks.
    2. A Databricks Vector Search index on that Delta table.
    3. A Knowledge Graph (vertices and edges) in Delta tables.
    """
    try:
        from databricks.vector_search.client import VectorSearchClient

        # Load and split documents
        documents = load_documents(docs_dir)
        if not documents:
            print("No documents found to process.")
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        splits = splitter.split_documents(documents)
        
        # --- Build Knowledge Graph ---
        # This should happen before modifying the splits for the vector index
        build_and_save_graph(splits)

        # --- Build Vector Search Index ---
        print("\n🏗️ Building Databricks Vector Search Index...")
        client = databricks_config.client
        spark = get_spark_session()
        
        # Prepare DataFrame for Delta table
        data = [{"id": i, "text": doc.page_content, "source": doc.metadata.get("source", "unknown")} for i, doc in enumerate(splits)]
        df = spark.createDataFrame(data)
        
        # Write to Delta table
        table_name = f"{databricks_config.catalog_name}.{databricks_config.schema_name}.rag_documents"
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)
        print(f"✅ Documents saved to Delta table: {table_name}")

        # Create the Vector Search index
        vsc = VectorSearchClient()
        endpoint_name = databricks_config.vector_search_endpoint
        index_name = databricks_config.vector_search_index_name
        
        print(f"Creating index '{index_name}' on endpoint '{endpoint_name}'...")
        vsc.create_delta_sync_index(
            endpoint_name=endpoint_name,
            source_table_name=table_name,
            index_name=index_name,
            pipeline_type="TRIGGERED",
            primary_key="id",
            embedding_source_column="text",
            embedding_model_endpoint_name="databricks-bge-large-en" # Using a Databricks Foundation Model for embeddings
        )
        print(f"✅ Vector Search index creation initiated.")

    except Exception as e:
        print(f"❌ Error building Databricks artifacts: {e}")
    finally:
        if 'spark' in locals() and spark.getActiveSession():
            spark.stop()

def build_index(docs_dir: str, index_dir: str, use_databricks: bool = False):
    """Build index using either FAISS or Databricks artifacts."""
    if use_databricks:
        build_databricks_artifacts(docs_dir)
    else:
        build_faiss_index(docs_dir, index_dir)

if __name__ == "__main__":
    docs = os.getenv("DOCS_DIR", "data/docs")
    idx = os.getenv("INDEX_DIR", "data/index/faiss")
    use_databricks = os.getenv("USE_DATABRICKS", "false").lower() == "true"
    
    build_index(docs, idx, use_databricks=use_databricks)
    if use_databricks:
        print("\nDatabricks ingestion process complete.")
    else:
        print(f"\nSaved FAISS index to: {idx}")