import os
import gradio as gr
from dotenv import load_dotenv

from src.rag.chain import create_rag_chain


load_dotenv()


INDEX_DIR = os.getenv("INDEX_DIR", "data/index/faiss")


def ensure_index_exists() -> None:
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(
            f"Index not found at {INDEX_DIR}. Run: python -m src.rag.ingest"
        )


def build_interface():
    ensure_index_exists()
    chain = create_rag_chain(INDEX_DIR)

    def answer_question(question: str) -> str:
        if not question or not question.strip():
            return "Please enter a question."
        return chain.invoke(question.strip())

    with gr.Blocks(title="Maverick RAG") as demo:
        gr.Markdown("# Maverick RAG — Llama-powered Retrieval QA")

        with gr.Row():
            inp = gr.Textbox(label="Ask a question", lines=2, placeholder="What does the charter specify for the agent framework?")
        with gr.Row():
            out = gr.Markdown(label="Answer")

        btn = gr.Button("Ask Maverick")
        btn.click(fn=answer_question, inputs=inp, outputs=out)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()


