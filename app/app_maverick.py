import gradio as gr
import traceback
from dotenv import load_dotenv

# --- Project Imports ---
from src.rag.chain import create_rag_chain, create_databricks_rag_chain
from src.utils.config import config_manager

# Load environment variables
load_dotenv()

# --- CSS for modern Gemini-like styling ---
CSS = """
/* Dark theme with modern fonts */
body, .gradio-container {
    background-color: #131314;
    color: #E3E3E3;
    font-family: 'Google Sans', sans-serif;
}
/* Main chat container */
#chat-window {
    height: 75vh;
    display: flex;
    flex-direction: column;
}
/* Center the main interface */
.gradio-container > .main, .gradio-container > .main > .wrap {
    max-width: 800px;
    margin: auto;
    padding-top: 2rem;
}
/* Chatbot styling */
#chatbot {
    background-color: #1E1F20;
    border: none;
    border-radius: 12px;
    box-shadow: none;
}
/* Text input area */
textarea {
    background-color: #1E1F20 !important;
    color: #E3E3E3 !important;
    border: 1px solid #444746 !important;
    border-radius: 24px !important;
    padding: 12px 16px !important;
}
/* Example prompts styling */
.gradio-examples {
    gap: 0.75rem !important;
}
.gradio-examples .example-button {
    background-color: #1E1F20 !important;
    color: #E3E3E3 !important;
    border: 1px solid #444746 !important;
    border-radius: 8px !important;
    text-align: left;
    transition: background-color 0.2s;
}
.gradio-examples .example-button:hover {
    background-color: #2F3031 !important;
}
/* Send button styling */
#submit-button {
    background-color: #1E1F20 !important;
    border: 1px solid #444746 !important;
    min-width: 50px !important;
    border-radius: 50% !important;
}
#submit-button:hover {
    background-color: #2F3031 !important;
}
/* Footer styling */
footer { display: none !important; }
"""

def build_maverick_interface():
    """
    Builds the Gradio ChatInterface and returns it.
    """
    try:
        print("Initializing RAG chain...")
        rag_chain = create_databricks_rag_chain() if config_manager.data.use_databricks else create_rag_chain(config_manager.data.index_dir)
        print("✅ RAG chain initialized successfully.")
    except Exception as e:
        print(f"❌ Error initializing RAG chain: {e}")
        rag_chain = None

    def predict(message, history):
        if rag_chain is None:
            yield "Error: RAG chain is not initialized. Please check the console for errors."
            return
        try:
            response = ""
            for chunk in rag_chain.stream(message):
                response += chunk
                yield response
        except Exception as e:
            print("--- An error occurred in the RAG chain ---")
            traceback.print_exc()
            print("------------------------------------------")
            yield f"An error occurred. Please check the application logs for details."

    # --- MODIFIED: Use gr.ChatInterface directly ---
    demo = gr.ChatInterface(
        fn=predict,
        title="Maverick RAG",
        description="Your enterprise conversational AI assistant",
        chatbot=gr.Chatbot(
            elem_id="chatbot",
            bubble_full_width=False,
            avatar_images=(None, "https://i.imgur.com/C75nCtx.png"),
        ),
        textbox=gr.Textbox(placeholder="Ask me anything about your documents...", container=False, scale=7),
        submit_btn=gr.Button("➤", elem_id="submit-button"),
        examples=[
            "What is the project charter about?",
            "How do I set up the environment?",
            "What technology stack is used in the project?",
        ],
        retry_btn=None,
        undo_btn=None,
        clear_btn=None,
        css=CSS,
        theme="NoCrypt/miku"
    )
    return demo

if __name__ == "__main__":
    maverick_app = build_maverick_interface()
    print("Launching Gradio interface...")
    maverick_app.launch(
        server_name=config_manager.app.server_name,
        server_port=config_manager.app.server_port,
        share=True
    )