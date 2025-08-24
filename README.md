# RAG-chatbot
A local Retrieval-Augmented Generation (RAG) chatbot that answers questions based on uploaded documents .

# RAG Chatbot

## Description
This project is a **Retrieval-Augmented Generation (RAG) Chatbot** that allows users to ask questions about documents (PDFs) and get context-aware answers. 
It combines a **document retrieval system** with a **language model** to provide precise responses based on the uploaded documents.

---

## Features
- Upload PDFs and extract their content.
- Ask questions related to the uploaded documents.
- Get accurate, context-aware answers using RAG.
- Simple CLI and/or web interface for interaction.
- Supports multiple document uploads.

---

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/rag-chatbot.git
   
2.Navigate to the project folder:
  cd rag-chatbot

3.Create a virtual environment (optional but recommended):
   python -m venv venv

4.Activate the virtual environment:
  Linux/Mac:
    source venv/bin/activate
  Windows:
    venv\Scripts\activate

5.Install dependencies:
   pip install -r requirements.txt

USAGE

1.CLI Version
   Run the command-line interface:
        python app_cli.py
Upload your PDF and start asking questions.
Type exit to quit.

2.Gradio/Web Version
  Run the Gradio interface:
     python app_gradio.py

Open the displayed URL in your browser.
Upload PDFs and interact with the chatbot.

Folder Structure
rag-chatbot/
├─ data/              # Uploaded PDFs
├─ rag.py             # RAG logic (retrieval + generation)
├─ app_cli.py         # Command-line interface
├─ app_gradio.py      # Web interface using Gradio
├─ requirements.txt   # Python dependencies
└─ README.md
