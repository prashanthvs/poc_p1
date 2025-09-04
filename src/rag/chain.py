import os
from pathlib import Path
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .llm_provider import get_llm


def load_retriever(index_dir: str):
    embedding_model = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    store = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return store.as_retriever(search_kwargs={"k": 4})


SYSTEM_PROMPT = (
    "You are Maverick, a helpful enterprise RAG assistant. "
    "Answer the user's question using the provided context. Be concise. "
    "If the answer is not contained in the context, say you don't know."
)


def create_rag_chain(index_dir: str):
    retriever = load_retriever(index_dir)
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


