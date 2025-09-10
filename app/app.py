# app/app.py

import os
import gradio as gr
from dotenv import load_dotenv

from src.rag.chain import create_rag_chain, create_databracks_rag_chain
from src.utils.config import config

# Load environment variables from .env file
load_dotenv()

# --- Custom CSS translated from new_mock.html ---
CUSTOM_CSS = """
/* --- General Styles --- */
body, .gradio-container {
    background-color: #020617; /* slate-950 */
    color: #e2e8f0; /* slate-200 */
    font-family: 'Inter', sans-serif;
}
footer, .main-div { background-color: transparent !important; }

/* --- Glass Card Effect --- */
.glass-card {
    background: rgba(15, 23, 42, 0.6); /* slate-900 with transparency */
    backdrop-filter: blur(10px);
    border: 1px solid rgba(51, 65, 85, 0.5); /* slate-700 with transparency */
    border-radius: 0.75rem !important; /* rounded-xl */
    padding: 1.5rem !important;
}

/* --- Glow Effects for Pathfinder --- */
.path-glow { box-shadow: 0 0 15px rgba(56, 189, 248, 0.4), 0 0 5px rgba(56, 189, 248, 0.6); }
.path-node-glow { box-shadow: 0 0 15px rgba(56, 189, 248, 0.6); }

/* --- Custom Scrollbar --- */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #1e293b; } /* slate-800 */
::-webkit-scrollbar-thumb { background: #475569; border-radius: 4px; } /* slate-600 */
::-webkit-scrollbar-thumb:hover { background: #64748b; } /* slate-500 */

/* --- Chatbot Specific Styles --- */
#chatbot {
    background-color: #0f172a; /* slate-900 */
    border-radius: 0.5rem !important;
    border: 1px solid #1e293b; /* slate-800 */
    color: #cbd5e1; /* slate-300 */
    height: 400px;
}
#chatbot .user-message, #chatbot .bot-message { border-radius: 0.5rem; }
#chatbot .user-message { background-color: #1e293b; } /* slate-800 */
#chatbot .bot-message { background-color: #334155; } /* slate-700 */

/* --- Input Area Styles --- */
.input-area {
    background: #0f172a !important; /* slate-900 */
    border-radius: 0.5rem !important;
    padding: 1rem;
    border: 1px solid #1e293b; /* slate-800 */
}
#prompt_input {
    background-color: #1e293b !important; /* slate-800 */
    border: 1px solid #334155 !important; /* slate-700 */
    color: #e2e8f0 !important; /* slate-200 */
    border-radius: 0.375rem !important;
}
.example-btn {
    background-color: #334155 !important; /* slate-700 */
    color: #cbd5e1 !important; /* slate-300 */
    border: none !important;
    border-radius: 9999px !important; /* rounded-full */
    transition: background-color 0.2s;
}
.example-btn:hover { background-color: #475569 !important; } /* slate-600 */
.send-button { min-width: 40px !important; }
"""

# --- SVG Icons used in the mock-up ---
SVG_ICONS = {
    "compass": """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-compass"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>""",
    "arrow_right": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-arrow-right"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>""",
    "git_branch": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-git-branch-plus"><path d="M6 3v12"/><path d="M18 9a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/><path d="M6 21a3 3 0 1 0 0-6 3 3 0 0 0 0 6z"/><path d="M15 6a9 9 0 0 0-9 9"/><path d="M18 15v6"/><path d="M21 18h-6"/></svg>""",
    "box_select": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-box-select"><rect width="20" height="20" x="2" y="2" rx="2"/><path d="M8 6h8"/><path d="M6 8v8"/><path d="M12 6v12"/><path d="M18 8v8"/><path d="M8 12h8"/></svg>""",
    "chart_bar": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-bar-chart-3"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>""",
    "sparkles": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-sparkles"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>""",
    "send": """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-send"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>"""
}

def ensure_index_exists() -> None:
    """Ensure either local index or Databricks configuration exists."""
    validation = config_manager.validate_config()
    if not validation["valid"]:
        raise RuntimeError(f"Configuration validation failed: {', '.join(validation['errors'])}")
    if not config_manager.data.use_databricks and not os.path.exists(config_manager.data.index_dir):
        raise RuntimeError(f"Index not found at {config_manager.data.index_dir}. Run: python -m src.rag.ingest")

def build_interface():
    ensure_index_exists()
    
    chain = create_databricks_rag_chain() if config_manager.data.use_databricks else create_rag_chain(config_manager.data.index_dir)

    def answer_question(message: str, history: list):
        if not message or not message.strip():
            yield "Please enter a question."
            return
        try:
            # Use chain.stream for a streaming response
            response = ""
            for chunk in chain.stream(message.strip()):
                response += chunk
                yield response
        except Exception as e:
            yield f"Error: {str(e)}"

    # --- Build Gradio Interface ---
    with gr.Blocks(theme=gr.themes.Default(primary_hue="sky"), css=CUSTOM_CSS, title="EDC AI Assistant") as demo:
        # Header Section
        gr.HTML("""
            <header style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 2rem;">
                <div>
                    <h1 style="font-size: 1.875rem; font-weight: bold; color: white;">EDC Gen AI Portal</h1>
                    <p style="color: #94a3b8;">A dynamic simulation of GenAI.</p>
                </div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="text-align: right;">
                        <p style="font-weight: 600; color: white;">John Doe</p>
                        <p style="font-size: 0.875rem; color: #94a3b8;">AI Specialist</p>
                    </div>
                    <img src="https://placehold.co/48x48/64748B/FFFFFF?text=JD" alt="User Avatar" style="width: 48px; height: 48px; border-radius: 9999px; border: 2px solid #475569;">
                </div>
            </header>
        """)

        # Main 3-column Grid
        with gr.Row():
            # Left Column (2/3 width)
            with gr.Column(scale=2):
                # --- Goal Definition Card ---
                with gr.Column(elem_classes=["glass-card"]):
                    gr.HTML(f"""
                        <div style="display: flex; align-items: flex-start; gap: 1rem;">
                            {SVG_ICONS['compass']}
                            <div>
                                <h2 style="font-size: 1.25rem; font-weight: bold; color: white; margin-bottom: 0.5rem;">Onboarding - 101</h2>
                                <p style="color: #94a3b8; margin-bottom: 1rem;">Your step-by-step guide to life in EDC.</p>
                                <div style="display: flex; align-items: center; gap: 1rem; background-color: rgba(2,6,23,0.8); padding: 1rem; border-radius: 0.5rem;">
                                    <span style="font-weight: 600; color: #94a3b8;">FROM: New Joiner</span>
                                    {SVG_ICONS['arrow_right']}
                                    <span style="font-weight: bold; color: #38bdf8;">TO: GenAI Expert</span>
                                </div>
                            </div>
                        </div>
                    """)
                # --- Pathfinder Card (Static Representation) ---
                with gr.Column(elem_classes=["glass-card"]):
                    gr.HTML(f"""
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            {SVG_ICONS['git_branch']}
                            <h2 style="font-size: 1.25rem; font-weight: bold; color: white;">The Pathfinder: Visualizing Your Future</h2>
                        </div>
                        <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1.5rem;">Your initial onboarding requires completion of the trainings mentioned below.</p>
                        <img src="https://i.imgur.com/gY23T5p.png" alt="Pathfinder UI Mockup" style="width: 100%; height: auto;">
                    """)
                # --- Assistant Module (Functional Chatbot) ---
                with gr.Column(elem_classes=["glass-card"]):
                    gr.HTML(f"""
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                            {SVG_ICONS['box_select']}
                            <h2 style="font-size: 1.25rem; font-weight: bold; color: white;">EDC AI Assistant</h2>
                        </div>
                        <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">Ask 'any' questions to find relevant, quick answers pertaining to EDC.</p>
                    """)
                    chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=False)
                    with gr.Column(elem_classes=["input-area"]):
                        with gr.Row():
                            prompt_input = gr.Textbox(placeholder="Ask anything...", show_label=False, container=False, scale=10)
                            send_btn = gr.Button(value=SVG_ICONS['send'], elem_classes=["send-button"], scale=1)
                        with gr.Row():
                            gr.Button("How to claim medical reimbursement?", elem_classes=["example-btn"]).click(lambda: "How to claim medical reimbursement?", outputs=prompt_input)
                            gr.Button("How to obtain my T4 statement?", elem_classes=["example-btn"]).click(lambda: "How to obtain my T4 statement?", outputs=prompt_input)

            # Right Column (1/3 width)
            with gr.Column(scale=1):
                # --- Your Profile Card ---
                with gr.Column(elem_classes=["glass-card"]):
                    gr.HTML(f"""
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                           {SVG_ICONS['chart_bar']}
                           <h2 style="font-size: 1.25rem; font-weight: bold; color: white;">Your Profile</h2>
                        </div>
                        <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1.5rem;">These reflect your onboarding profile completion status.</p>
                        <img src="https://i.imgur.com/L7XT7M1.png" alt="Profile UI Mockup" style="width: 100%; height: auto;">
                    """)
                # --- Onboarding Support Simulator Card ---
                with gr.Column(elem_classes=["glass-card"]):
                    gr.HTML(f"""
                        <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                           {SVG_ICONS['sparkles']}
                           <h2 style="font-size: 1.25rem; font-weight: bold; color: white;">Onboarding Support Simulator</h2>
                        </div>
                        <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1.5rem;">Personalized, actionable recommendations to help you close your skill gaps.</p>
                        <img src="https://i.imgur.com/8Qp42bU.png" alt="Simulator UI Mockup" style="width: 100%; height: auto;">
                    """)

        # Connect Chatbot to backend function
        prompt_input.submit(answer_question, [prompt_input, chatbot], chatbot)
        send_btn.click(answer_question, [prompt_input, chatbot], chatbot)

    return demo

if __name__ == "__main__":
    maverick_app = build_interface()
    maverick_app.launch(
        server_name=config_manager.app.server_name,
        server_port=config_manager.app.server_port,
        share=False
    )