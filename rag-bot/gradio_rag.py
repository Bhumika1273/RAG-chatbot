# gradio_rag.py
import os
import time
import numpy as np
import subprocess
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Config
# -----------------------------
DB_DIR = "db_gradio"  # separate DB for the Gradio app
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_MODEL_MAP = {
    "mistral": "mistral:latest",
    "llama3": "llama3:latest",
    "llama2": "llama2:7b",
    "gemma":   "gemma:2b",
}
CHUNK_SIZE = 350
CHUNK_OVERLAP = 40
TOP_K = 3
SIM_THRESHOLD = 0.5  # cosine similarity cutoff for citing chunks

# -----------------------------
# Globals
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
vectorstore: FAISS | None = None

# Load index if present
if os.path.exists(DB_DIR):
    try:
        vectorstore = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
    except Exception:
        vectorstore = None

# -----------------------------
# Helpers
# -----------------------------
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def _get_llm(model_key: str) -> ChatOllama:
    """Automatically handle missing models, pull if needed, and fallback."""
    preferred_models = [model_key, "mistral", "llama2", "gemma"]  # fallback order

    for mk in preferred_models:
        tag = DEFAULT_MODEL_MAP.get(mk, mk)  # e.g., "mistral:latest"
        try:
            # Check if model exists locally
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if tag not in result.stdout:
                print(f"Pulling model {tag} automatically...")
                subprocess.run(["ollama", "pull", tag], check=True)
            
            # Initialize the model
            llm = ChatOllama(model=tag)
            print(f"Using model: {tag}")
            return llm
        except Exception as e:
            print(f"Failed to load model {tag}: {e}")
            continue
    
    raise RuntimeError("No Ollama model could be loaded!")

def _get_retriever():
    global vectorstore
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})

def _filter_relevant_chunks(query: str, docs) -> List[str]:
    """Return only chunk texts that pass cosine SIM_THRESHOLD against the query."""
    q_emb = embedding_model.embed_query(query)
    kept = []
    for d in docs or []:
        d_emb = embedding_model.embed_query(d.page_content)
        sim = _cosine_similarity(np.array(q_emb), np.array(d_emb))
        if sim >= SIM_THRESHOLD and d.page_content.strip():
            kept.append(d.page_content)
    return kept

# -----------------------------
# Public API
# -----------------------------
def ingest_new_file(file_path: str) -> dict:
    """Ingest a PDF into the Gradio vectorstore (db_gradio). Returns timings and counts."""
    t0 = time.time()
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    t1 = time.time()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    t2 = time.time()

    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(DB_DIR)
    t3 = time.time()

    return {
        "filename": os.path.basename(file_path),
        "pages": len(docs),
        "chunks": len(chunks),
        "t_load": round(t1 - t0, 2),
        "t_split": round(t2 - t1, 2),
        "t_save": round(t3 - t2, 2),
        "total": round(t3 - t0, 2),
    }

def ask_with_context(query: str, model: str = "mistral") -> Tuple[str, List[str]]:
    """Answer a question using the Gradio index. Returns (answer, cited_chunks)."""
    retriever = _get_retriever()
    if retriever is None:
        return "⚠️ Please upload a PDF first.", []

    llm = _get_llm(model)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    raw_answer = (result.get("result") or "").strip()
    source_docs = result.get("source_documents", [])
    cited_texts = _filter_relevant_chunks(query, source_docs)

    if not cited_texts:
        return "❌ Sorry, I cannot find an answer in the uploaded PDF.", []

    if not raw_answer:
        raw_answer = "I found related context, but couldn't compose an exact answer."

    return raw_answer, cited_texts

def reset_index():
    """Completely clears the Gradio vectorstore."""
    global vectorstore
    vectorstore = None
    if os.path.exists(DB_DIR):
        for fn in os.listdir(DB_DIR):
            try:
                os.remove(os.path.join(DB_DIR, fn))
            except Exception:
                pass
        try:
            os.rmdir(DB_DIR)
        except Exception:
            pass
