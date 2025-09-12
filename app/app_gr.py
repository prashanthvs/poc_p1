import gradio as gr

custom_css = """
body {
    font-family: 'Inter', sans-serif;
    background-color: #020617; /* slate-950 */
    color: #e2e8f0; /* slate-200 */
}
.gradio-container {
    max-width: 1400px !important;
}
.glass-card {
    background: rgba(15, 23, 42, 0.6); /* slate-900 with transparency */
    backdrop-filter: blur(10px);
    border: 1px solid rgba(51, 65, 85, 0.5); /* slate-700 */
    border-radius: 12px;
    padding: 16px;
}
.path-node {
    text-align: center;
    font-size: 12px;
    color: #e2e8f0;
}
.path-connector {
    flex: 1;
    height: 4px;
    background-color: #38bdf8; /* sky-400 */
    border-radius: 9999px;
}
.badge-btn {
    background: rgba(51,65,85,0.5);
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 12px;
    cursor: pointer;
}
.badge-btn:hover {
    background: rgba(71,85,105,0.9);
}
"""

def sandbox_response(query):
    # Simulated AI assistant response
    return f"🔹 Simulation Result: For query '{query}', follow the standard EDC process."

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Row():
        with gr.Column(scale=2, elem_classes="glass-card"):
            gr.Markdown("## 🎯 Onboarding - 101\nYour step-by-step guide to life in EDC.")
            gr.Markdown("FROM: **New Joiner** → TO: **GenAI Expert**")
            gr.Button("Change Goal")
            
            # Horizontal Separator
            gr.HTML("<hr style='border: 0; height: 1px; background-color: #334155; margin: 20px 0;'>")

            gr.Markdown("## 🧭 The Pathfinder: Visualizing Your Future\nYour initial onboarding requires completion of the trainings mentioned below.")
            with gr.Row():
                gr.HTML("<div class='path-node'>📌 You are here<br/>HR Onboarding</div>")
                gr.HTML("<div class='path-connector'></div>")
                gr.HTML("<div class='path-node'>⏭ Next Step<br/>EDC Benefits</div>")
                gr.HTML("<div class='path-connector'></div>")
                gr.HTML("<div class='path-node'>📂 Training #2<br/>Cybersecurity 101</div>")
                gr.HTML("<div class='path-connector'></div>")
                gr.HTML("<div class='path-node'>🏁 Training #3<br/>Ethics 101</div>")
                
            # Horizontal Separator
            gr.HTML("<hr style='border: 0; height: 1px; background-color: #334155; margin: 20px 0;'>")

            gr.Markdown("## 🤖 EDC AI Assistant\nAsk any questions to find relevant, quick answers pertaining to EDC.")
            query = gr.Textbox(placeholder="e.g., Ask anything?", show_label=False)
            example_btns = gr.Row([gr.Button("How to claim medical reimbursement?", elem_classes="badge-btn"),
                                   gr.Button("How to obtain my T4 statement?", elem_classes="badge-btn"),
                                   gr.Button("Guide me to raise an IT ticket for a new laptop purchase.", elem_classes="badge-btn")])
            response = gr.Markdown()
            query.submit(sandbox_response, query, response)
            for b in example_btns.children:
                b.click(lambda x=b.value: sandbox_response(x), None, response)

        with gr.Column(scale=1, elem_classes="glass-card"):
            gr.Markdown("## 📊 Your Profile\nThese reflect your onboarding profile completion status.")
            with gr.Row():
                gr.Label("Trainings: 80%")
            with gr.Row():
                gr.Label("Profile: 100%")
            with gr.Row():
                gr.Label("Appraisal: 45%")

            gr.Markdown("## ✨ Onboarding Support Simulator\nPersonalized, actionable recommendations to help you close your skill gaps and advance.")
            gr.Markdown("👥 **Find a Mentor** – Connect with Frank Miller for leadership guidance.")
            gr.Markdown("🚀 **Join a Project** – 'Project Phoenix' needs a marketing lead.")
            gr.Markdown("📖 **Acquire Knowledge** – Read case study on 'EDC project accelerator'.")
            
demo.launch()
