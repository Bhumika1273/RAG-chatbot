# ingest.py
import os
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = "data"
DB_DIR = "db"

print("=== PDF Ingestion Started ===\n")

# Step 1: Load PDFs
t0 = time.time()
documents = []
for file_name in os.listdir(DATA_DIR):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_DIR, file_name))
        docs = loader.load()
        documents.extend(docs)
t1 = time.time()
load_time = round(t1 - t0, 3)
print(f"Step 1: Loaded {len(documents)} pages from PDFs in {load_time} seconds")

# Step 2: Split into chunks
chunk_size = 300
chunk_overlap = 30
t2 = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
chunks = text_splitter.split_documents(documents)
t3 = time.time()
chunk_time = round(t3 - t2, 3)
print(f"Step 2: Split into {len(chunks)} chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap}) in {chunk_time} seconds")

# Step 3: Generate embeddings
t4 = time.time()
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embedding_model)
t5 = time.time()
embedding_time = round(t5 - t4, 3)
print(f"Step 3: Generated embeddings for {len(chunks)} chunks in {embedding_time} seconds")

# Step 4: Save vectorstore
t6 = time.time()
vectorstore.save_local(DB_DIR)
t7 = time.time()
save_time = round(t7 - t6, 3)
print(f"Step 4: Saved vectorstore to '{DB_DIR}' in {save_time} seconds")

# Show DB size
loaded_vs = FAISS.load_local(DB_DIR, embedding_model, allow_dangerous_deserialization=True)
print(f"\n✅ Vectorstore contains {loaded_vs.index.ntotal} embeddings (vectors)")
print("\n=== PDF Ingestion Completed ===")

# Total time
total_time = round((t7 - t0), 3)
print(f"\nTotal ingestion time: {total_time} seconds")
