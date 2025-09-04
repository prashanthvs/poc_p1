import os
from pathlib import Path
from typing import List, Optional
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, monotonically_increasing_id
from delta.tables import DeltaTable

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..utils.databricks_config import databricks_config


def load_documents(input_dir: str) -> List:
    """Load documents from either local directory or Unity Catalog volume."""
    directory = Path(input_dir)
    
    # Check if we're using Unity Catalog volume
    if input_dir.startswith("/Volumes/"):
        return load_documents_from_volume(input_dir)
    
    # Local directory loading
    if not directory.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Load common text-like files
    loader = DirectoryLoader(
        input_dir,
        glob="**/*",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def load_documents_from_volume(volume_path: str) -> List:
    """Load documents from Unity Catalog volume using Spark."""
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("RAG Document Ingestion") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        # Read files from volume
        df = spark.read.text(volume_path)
        
        # Convert to LangChain documents
        documents = []
        for row in df.collect():
            # Create a simple document from text content
            from langchain_core.documents import Document
            doc = Document(
                page_content=row.value,
                metadata={"source": volume_path}
            )
            documents.append(doc)
        
        spark.stop()
        return documents
        
    except Exception as e:
        print(f"Error loading documents from volume {volume_path}: {e}")
        # Fallback to empty list
        return []


def build_faiss_index(docs_dir: str, index_dir: str) -> str:
    """Build FAISS index (fallback for local development)."""
    documents = load_documents(docs_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        add_start_index=True,
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


def build_databricks_vector_index(docs_dir: str) -> str:
    """Build vector search index in Databricks with managed embeddings."""
    try:
        # Ensure Unity Catalog volume exists
        databricks_config.create_volume_if_not_exists()
        
        # Load documents
        documents = load_documents(docs_dir)
        
        if not documents:
            print("No documents found to process")
            return ""
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits = splitter.split_documents(documents)
        
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("RAG Vector Index Creation") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        # Prepare data for vector search
        data = []
        for i, doc in enumerate(splits):
            data.append({
                "id": str(i),
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "chunk_index": doc.metadata.get("start_index", 0)
            })
        
        # Create DataFrame
        df = spark.createDataFrame(data)
        
        # Save to Delta table
        table_path = f"{databricks_config.catalog_name}.{databricks_config.schema_name}.rag_documents"
        df.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable(table_path)
        
        print(f"Documents saved to Delta table: {table_path}")
        
        # Note: Vector search index creation would be done through Databricks UI or API
        # This is a placeholder for the actual vector search index creation
        print(f"Vector search index should be created for table: {table_path}")
        print("Use Databricks Vector Search UI to create the index with managed embeddings")
        
        spark.stop()
        return table_path
        
    except Exception as e:
        print(f"Error building Databricks vector index: {e}")
        return ""


def build_index(docs_dir: str, index_dir: str, use_databricks: bool = False) -> str:
    """Build index using either FAISS (local) or Databricks vector search."""
    if use_databricks:
        return build_databricks_vector_index(docs_dir)
    else:
        return build_faiss_index(docs_dir, index_dir)


if __name__ == "__main__":
    docs = os.getenv("DOCS_DIR", "data/docs")
    idx = os.getenv("INDEX_DIR", "data/index/faiss")
    use_databricks = os.getenv("USE_DATABRICKS", "false").lower() == "true"
    
    if use_databricks:
        # Use Unity Catalog volume if specified
        volume_path = os.getenv("DATABRICKS_VOLUME_PATH")
        if volume_path:
            docs = volume_path
        
        path = build_index(docs, idx, use_databricks=True)
        print(f"Built Databricks vector index: {path}")
    else:
        path = build_index(docs, idx, use_databricks=False)
        print(f"Saved FAISS index to: {path}")


