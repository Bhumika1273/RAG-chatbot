import os
import gradio as gr
import time
from rag import qa_chain, ingest_new_file 

uploaded_files = []  # Keeping track of uploaded PDFs

# Function to handle question queries
def ask_question_ui(query):
    start_time = time.time()
    result = qa_chain({"query": query})
    end_time = time.time()
    latency = round(end_time - start_time, 2)

    # Answer
    answer = f"""
    <div style="font-size:24px; font-weight:600; line-height:1.6; color:white; margin-bottom:15px;">
        <b>Answer:</b><br>{result['result']}
        <br><br><span style="font-size:16px; color:#ccc;">⏱ Latency: {latency} seconds</span>
    </div>
    """

    # Chunks 
    sources = "<div style='font-size:20px; color:white; line-height:1.5; margin-top:10px;'>"
    sources += "<b>📑 Cited Chunks:</b><br><br>"
    for i, doc in enumerate(result['source_documents']):
        sources += f"<div style='margin-bottom:12px; color:white;'>"
        sources += f"<b>Chunk {i+1}:</b> {doc.page_content}</div>"
    sources += "</div>"

    return answer + sources 

# Function to handle PDF upload
def upload_pdf(files):
    global uploaded_files
    messages = []
    for file in files:
        filename = os.path.basename(file.name)
        ingest_new_file(file.name)
        uploaded_files.append(filename)
        
        messages.append( f"<div style='background-color:#d4edda; color:#155724; padding:10px; " f"border-radius:8px; margin-bottom:8px; font-size:18px; font-weight:bold;'>" f"✅ {filename} ingested successfully!" f"</div>" )
    uploaded_list = "<ul style='font-size:18px; color:white; line-height:1.6;'>" + "".join([f"<li>{f}</li>" for f in uploaded_files]) + "</ul>"
    return "".join(messages) + f"<h4 style='font-size:20px; color:white;'>📂 Uploaded PDFs:</h4>{uploaded_list}"

# Gradio interface
with gr.Blocks(css="""
    /* ✅ Make buttons smaller */
    .gr-button {
        background-color: #4CAF50; 
        color: white; 
        font-weight: bold;
        width: 200px !important;
        font-size: 18px; 
        padding: 10px; 
        border-radius: 8px;
    }

    /* ✅ Bigger input + white placeholder */
    textarea, input {
        font-size: 22px !important; 
        font-weight: 500; 
        line-height: 1.5; 
        color: white !important;
        background-color: #222 !important;
    }
    textarea::placeholder, input::placeholder {
        font-size: 20px;
        color: #bbb;
    }
""") as demo:
    
    gr.Markdown("<h1 style='color:white;'>📚 RAG Bot - Ask Questions on Your PDFs</h1>")
    gr.Markdown("<p style='color:#ddd;'>Upload your PDFs first, then ask questions. Answers will include cited chunks from your documents.</p>")

    pdf_input = gr.File(file_types=[".pdf"], file_count="multiple")
    upload_btn = gr.Button("Upload PDF(s)")
    upload_msg = gr.HTML()  
    upload_btn.click(upload_pdf, inputs=pdf_input, outputs=upload_msg)

    query_input = gr.Textbox(label="Your Question", placeholder="Type your question here...", lines=2)
    ask_btn = gr.Button("Get Answer")
    output_display = gr.HTML(label="Answer + Chunks") 
    ask_btn.click(ask_question_ui, inputs=query_input, outputs=output_display)

demo.launch()
