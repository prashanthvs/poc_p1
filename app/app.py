import os
import gradio as gr
from dotenv import load_dotenv

from src.rag.chain import create_rag_chain, create_databricks_rag_chain
from src.utils import config_manager


load_dotenv()


def ensure_index_exists() -> None:
    """Ensure either local index or Databricks configuration exists."""
    # Validate configuration
    validation = config_manager.validate_config()
    
    if not validation["valid"]:
        raise RuntimeError(f"Configuration validation failed: {', '.join(validation['errors'])}")
    
    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"Warning: {warning}")
    
    # Check specific requirements based on configuration
    if config_manager.data.use_databricks:
        # Databricks configuration is validated above
        pass
    else:
        # Check local index
        if not os.path.exists(config_manager.data.index_dir):
            raise RuntimeError(
                f"Index not found at {config_manager.data.index_dir}. Run: python -m src.rag.ingest"
            )


def build_interface():
    ensure_index_exists()
    
    # Create appropriate chain based on configuration
    if config_manager.data.use_databricks:
        chain = create_databricks_rag_chain()
        title = "Maverick RAG — Databricks-powered Retrieval QA"
    else:
        chain = create_rag_chain(config_manager.data.index_dir)
        title = "Maverick RAG — Llama-powered Retrieval QA"

    def answer_question(question: str) -> str:
        if not question or not question.strip():
            return "Please enter a question."
        try:
            return chain.invoke(question.strip())
        except Exception as e:
            return f"Error: {str(e)}"

    with gr.Blocks(title=config_manager.app.title) as demo:
        gr.Markdown(f"# {title}")
        
        # Show configuration info
        config_summary = config_manager.get_config_summary()
        if config_manager.data.use_databricks:
            gr.Markdown(f"""
            **Configuration:**
            - Workspace: {config_summary['databricks']['workspace_url']}
            - Catalog: {config_summary['databricks']['catalog']}
            - Schema: {config_summary['databricks']['schema']}
            - Volume: {config_summary['databricks']['volume']}
            - LLM Provider: {config_summary['llm']['provider']}
            - Model: {config_summary['llm']['model']}
            """)
        else:
            gr.Markdown(f"""
            **Configuration:**
            - LLM Provider: {config_summary['llm']['provider']}
            - Model: {config_summary['llm']['model']}
            - Embedding Model: {config_summary['embedding']['model']}
            - Documents Directory: {config_summary['data']['docs_dir']}
            """)

        with gr.Row():
            inp = gr.Textbox(label="Ask a question", lines=2, placeholder="What does the charter specify for the agent framework?")
        with gr.Row():
            out = gr.Markdown(label="Answer")

        btn = gr.Button("Ask Maverick")
        btn.click(fn=answer_question, inputs=inp, outputs=out)

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name=config_manager.app.server_name,
        server_port=config_manager.app.server_port
    )


