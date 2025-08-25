import os
import time
import gradio as gr
from langchain.memory import ConversationBufferMemory
from gradio_rag import ask_with_context, ingest_new_file
import io
import sys

# Memory Setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tracking uploaded files
uploaded_files = []

# Model switching
AVAILABLE_MODELS = ["mistral", "llama2", "gemma"]
current_model = "mistral"  # default

def switch_model(model_name):
    global current_model
    if model_name in AVAILABLE_MODELS:
        current_model = model_name
        return f"✅ Switched to **{model_name}**"
    return f"⚠️ Model {model_name} not available."

# Displaying chat history
def format_chat_history(history):
    if not history:
        return "_No chat history yet._"
    formatted = ""
    for i, msg in enumerate(history):
        role = "👤 You" if i % 2 == 0 else "🤖 Bot"
        formatted += f"<div style='font-size:18px; color:white; margin-bottom:8px;'><b>{role}:</b> {msg.content}</div>"
    return formatted

# Handle PDF upload
def upload_pdf(files):
    global uploaded_files
    messages = []
    uploaded_files = []  # resetting list
    for file in files:
        ingest_new_file(file.name)
        short_name = os.path.basename(file.name)
        uploaded_files.append(short_name)
        messages.append(f"<div style='color:white; font-size:18px; margin-bottom:6px;'>📄 {short_name} ingested successfully!</div>")
    uploaded_list = "".join([f"<div style='color:white; font-size:18px;'>- {f}</div>" for f in uploaded_files])
    return "".join(messages), f"<b style='color:white; font-size:20px;'>📂 Uploaded PDFs:</b><br>{uploaded_list}"

# Handling Question + Memory + Model
def ask_question_ui(query):
    start_time = time.time()

    try:
        answer_text, cited_chunks = ask_with_context(query, model=current_model)
    except Exception as e:
        answer_text = f"❌ Error generating answer: {str(e)}"
        cited_chunks = []

    # Save question + answer into memory
    memory.save_context({"input": query}, {"output": answer_text})

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    # Answer formatting
    output_html = f"""
    <div style="font-size:24px; font-weight:600; line-height:1.6; color:white;">
        {answer_text.replace('\n', '<br>')}
        <br><span style="font-size:16px; color:#ccc;">⏱ Latency: {latency} seconds | Model: {current_model}</span>
    </div>
    """

    # Expandable cited chunks
    if cited_chunks:
        output_html += "<h4 style='color:white; margin-top:15px;'>📑 Cited Chunks:</h4>"
        for i, chunk in enumerate(cited_chunks, 1):
            output_html += f"""
            <details style="margin-bottom:8px;">
                <summary style="color:#4CAF50; font-size:18px;">📄 Chunk {i}</summary>
                <div style="color:white; font-size:16px; margin-left:15px;">{chunk}</div>
            </details>
            """

    history = memory.load_memory_variables({}).get("chat_history", [])
    return output_html, format_chat_history(history)

# Gradio Interface
with gr.Blocks(css="""
    .gr-button {background-color: #4CAF50; color: white; font-weight: bold; width: 200px !important; font-size: 18px; padding: 10px; border-radius: 8px;}
    textarea, input {font-size: 22px !important; font-weight: 500; line-height: 1.5; color: white !important; background-color: #222 !important;}
    textarea::placeholder, input::placeholder {font-size: 20px; color: #bbb;}
    .gr-markdown, .gr-html {font-size: 20px !important; color: white !important;}
""") as demo:

    gr.Markdown("<h1 style='color:white;'>🚀 RAG Bot </h1>")
    gr.Markdown("<p style='color:#ccc;'>Includes chat memory, model switching, cited context, and dynamic PDF ingestion.</p>")

    # Model selection
    model_dropdown = gr.Dropdown(choices=AVAILABLE_MODELS, value=current_model, label="Choose Model")
    model_status = gr.Markdown()
    model_dropdown.change(switch_model, inputs=model_dropdown, outputs=model_status)

    # File upload
    pdf_input = gr.File(file_types=[".pdf"], file_count="multiple")
    upload_btn = gr.Button("Upload PDF(s)")
    upload_msg = gr.HTML()
    uploaded_list_md = gr.HTML()
    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=[upload_msg, uploaded_list_md])

    # Ask questions
    query_input = gr.Textbox(label="Your Question", placeholder="Ask me anything from the PDFs...", lines=2)
    ask_btn = gr.Button("Get Answer")
    answer_output = gr.HTML(label="Answer + Cited Chunks")
    
    # Chat history
    chat_history_heading = gr.Markdown("<h3 style='color:white;'>💬 Chat History</h3>")
    history_output = gr.HTML()

    ask_btn.click(ask_question_ui, inputs=query_input, outputs=[answer_output, history_output])

demo.launch()
