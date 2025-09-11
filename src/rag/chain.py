# src/rag/chain.py

import os
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS, DatabricksVectorSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .llm_provider import get_llm
from .graph import GraphRAGRetriever
# CORRECTED IMPORT
from utils.config import config_manager


def load_retriever(index_dir: str, use_databricks: bool = False):
    """Load retriever from either FAISS (local) or Databricks vector search."""
    if use_databricks:
        # --- FIXED: Using actual DatabricksVectorSearch ---
        from databricks.vector_search.client import VectorSearchClient
        vsc = VectorSearchClient()
        index = vsc.get_index(
            # CORRECTED CONFIGURATION ACCESS
            endpoint_name=config_manager.databricks.vector_search_endpoint,
            index_name=config_manager.databricks.vector_search_index_name
        )
        # Using a Databricks foundation model for query embeddings
        dvs = DatabricksVectorSearch(
            index,
            embedding_model_endpoint_name="databricks-bge-large-en",
            text_column="text"
        )
        return dvs.as_retriever(search_kwargs={"k": 4})
    else:
        return load_faiss_retriever(index_dir)

def load_faiss_retriever(index_dir: str):
    """Load FAISS retriever (fallback for local development)."""
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return store.as_retriever(search_kwargs={"k": 4})

SYSTEM_PROMPT = (
    "You are Maverick, a helpful enterprise RAG assistant. "
    "Answer the user's question using only the provided context. Be concise. "
    "If the answer is not contained in the context, say you don't know."
)

def format_docs(docs: List[Document]) -> str:
    """Format documents for the RAG chain context."""
    chunks = []
    for d in docs:
        meta = d.metadata or {}
        source = meta.get("source", "N/A")
        chunks.append(f"\n{d.page_content}")
    return "\n\n".join(chunks)

def create_rag_chain(index_dir: str, use_databricks: bool = False):
    """Create a standard RAG chain using vector search."""
    retriever = load_retriever(index_dir, use_databricks)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {question}\n\nContext:\n{context}\n\nProvide an answer and cite sources."),
    ])

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def create_graph_rag_chain():
    """Create a RAG chain that uses the Knowledge Graph retriever."""
    retriever = GraphRAGRetriever()
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on a knowledge graph. "
                   "The context provided contains relationships between entities. Use it to answer the question."),
        ("human", "Question: {question}\n\nContext from Knowledge Graph:\n{context}\n\nAnswer:"),
    ])

    graph_rag_chain = (
        {"context": retriever.get_relevant_documents, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return graph_rag_chain

def create_databricks_rag_chain():
    """Convenience function for creating the Databricks RAG chain."""
    return create_rag_chain("", use_databricks=True)