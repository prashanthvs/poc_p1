import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_documents(input_dir: str) -> List:
    directory = Path(input_dir)
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


def build_faiss_index(docs_dir: str, index_dir: str) -> str:
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


if __name__ == "__main__":
    docs = os.getenv("DOCS_DIR", "data/docs")
    idx = os.getenv("INDEX_DIR", "data/index/faiss")
    path = build_faiss_index(docs, idx)
    print(f"Saved FAISS index to: {path}")


