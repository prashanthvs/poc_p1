import os
from pathlib import Path
from typing import Dict, List, Optional
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .llm_provider import get_llm
from ..utils.databricks_config import databricks_config


def load_retriever(index_dir: str, use_databricks: bool = False):
    """Load retriever from either FAISS (local) or Databricks vector search."""
    if use_databricks:
        return load_databricks_retriever()
    else:
        return load_faiss_retriever(index_dir)


def load_faiss_retriever(index_dir: str):
    """Load FAISS retriever (fallback for local development)."""
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return store.as_retriever(search_kwargs={"k": 4})


def load_databricks_retriever():
    """Load retriever from Databricks vector search with managed embeddings."""
    class DatabricksRetriever:
        def __init__(self):
            self.vector_search_endpoint = databricks_config.vector_search_endpoint
            self.index_name = databricks_config.vector_search_index_name
            
        def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
            """Retrieve relevant documents using Databricks vector search."""
            try:
                # This is a placeholder for the actual vector search API call
                # In practice, you would use the Databricks Vector Search API
                return self._query_vector_search(query, k)
            except Exception as e:
                print(f"Error querying vector search: {e}")
                return []
        
        def _query_vector_search(self, query: str, k: int) -> List[Document]:
            """Query Databricks vector search endpoint."""
            # Placeholder implementation - would use actual Databricks Vector Search API
            # For now, return empty list
            print(f"Querying vector search for: {query} (k={k})")
            return []
        
        def __call__(self, query: str) -> List[Document]:
            return self.get_relevant_documents(query)
    
    return DatabricksRetriever()


SYSTEM_PROMPT = (
    "You are Maverick, a helpful enterprise RAG assistant. "
    "Answer the user's question using the provided context. Be concise. "
    "If the answer is not contained in the context, say you don't know."
)


def create_rag_chain(index_dir: str, use_databricks: bool = False):
    """Create RAG chain using either local FAISS or Databricks vector search."""
    retriever = load_retriever(index_dir, use_databricks)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question: {question}\n\nContext:\n{context}\n\nProvide an answer and cite sources.",
            ),
        ]
    )

    def format_docs(docs: List) -> str:
        chunks = []
        for d in docs:
            meta = d.metadata or {}
            source = meta.get("source")
            chunks.append(f"[source: {source}]\n{d.page_content}")
        return "\n\n".join(chunks)

    retriever_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever_chain


def create_databricks_rag_chain():
    """Create RAG chain specifically for Databricks deployment."""
    return create_rag_chain("", use_databricks=True)


