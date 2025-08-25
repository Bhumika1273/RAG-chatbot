import os
import time
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# -----------------------------
# Embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# Load or create vectorstore
# -----------------------------
vectorstore = None
if os.path.exists("db"):
    vectorstore = FAISS.load_local("db", embedding_model, allow_dangerous_deserialization=True)
    print(f"✅ Loaded existing vectorstore with {vectorstore.index.ntotal} embeddings")
else:
    print("⚠️ No existing vectorstore found. It will be created from PDFs in 'data/' folder.")

# -----------------------------
# Cosine similarity
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -----------------------------
# Create retriever
# -----------------------------
def create_retriever_from_chunks(chunks):
    temp_vs = FAISS.from_documents(chunks, embedding_model)
    return temp_vs.as_retriever(search_kwargs={"k": 1})  # top 3 chunks

# -----------------------------
# Create QA chain
# -----------------------------
def create_qa_chain(retriever):
    return RetrievalQA.from_chain_type(
        llm=ChatOllama(model="mistral:latest"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# -----------------------------
# Ask function
# -----------------------------
def ask(query, qa_chain):
    start_time = time.time()
    result = qa_chain.invoke({"query": query})
    latency = round(time.time() - start_time, 2)

    answer = result.get('result', '').strip()
    source_docs = result.get('source_documents', [])

    # Filter chunks using cosine similarity
    relevant_chunks = []
    query_embedding = embedding_model.embed_query(query)
    for doc in source_docs:
        doc_embedding = embedding_model.embed_query(doc.page_content)
        similarity = cosine_similarity(query_embedding, doc_embedding)
        if similarity >= 0.5 and doc.page_content.strip():
            relevant_chunks.append(doc)

    if not answer or not relevant_chunks:
        print("\n❌ Sorry, I cannot find an answer in the PDFs.")
    else:
        print("\n--- Answer ---")
        print(answer)
        print(f"\n⏱ Latency: {latency} seconds")

        print("\n--- Cited Chunks ---")
        for i, doc in enumerate(relevant_chunks):
            print(f"Chunk {i+1}: {doc.page_content}\n------")

# -----------------------------
# Ingest PDF
# -----------------------------
def ingest_new_file(file_path):
    t0 = time.time()
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    t1 = time.time()

    chunk_size = 300
    chunk_overlap = 30
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    t2 = time.time()

    global vectorstore
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embedding_model)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local("db")
    t3 = time.time()

    print(f"📄 {os.path.basename(file_path)} ingested successfully! Pages: {len(docs)}, chunks: {len(chunks)}")
    return create_retriever_from_chunks(chunks)

# -----------------------------
# Auto-ingest all PDFs in 'data/'
# -----------------------------
data_folder = "data"
pdf_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]
if not pdf_files:
    print(f"⚠️ No PDFs found in '{data_folder}' folder. Please add files.")
else:
    print(f"📂 Found {len(pdf_files)} PDF(s) in '{data_folder}', ingesting...")
    for f in pdf_files:
        retriever = ingest_new_file(f)
    qa_chain = create_qa_chain(retriever)
    print("✅ All PDFs ingested and QA chain ready.")

# -----------------------------
# CLI loop
# -----------------------------
if __name__ == "__main__":
    print("\n🚀 RAG CLI Ready! Ask questions (type 'exit' to quit).")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if 'qa_chain' not in locals():
            print("⚠️ No PDFs ingested. Please add PDFs to the 'data/' folder.")
            continue
        ask(query, qa_chain)
