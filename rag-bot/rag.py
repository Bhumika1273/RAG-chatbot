import os
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load existing vectorstore
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("db", embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOllama(model="mistral:latest"),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

def ask(query):
    start_time = time.time()
    result = qa_chain.invoke({"query": query})
    latency = round(time.time() - start_time, 2)

    # Show answer
    print("\n--- Answer ---")
    print(result['result'])
    print(f"\n⏱ Latency: {latency} seconds")

    # Only show cited chunks if relevance > threshold
    if result['source_documents']:
        print("\n--- Cited Chunks ---")
        for i, doc in enumerate(result['source_documents']):
            print(f"Chunk {i+1}: {doc.page_content}\n------")
    else:
        print("\n⚠️ Out of context: No relevant documents found.")

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
    vectorstore.add_documents(chunks)
    vectorstore.save_local("db")
    t3 = time.time()

    print(f"{os.path.basename(file_path)} ingested successfully!")
    print(f"📄 Pages loaded: {len(docs)} (took {round(t1 - t0, 2)} s)")
    print(f"🔹 Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap}) in {round(t2 - t1, 2)} s")
    print(f"⚡ Embeddings added and saved in {round(t3 - t2, 2)} s")

    # Show current vectorstore size
    loaded_vs = FAISS.load_local("db", embedding_model, allow_dangerous_deserialization=True)
    print(f"📦 Vectorstore now contains {loaded_vs.index.ntotal} embeddings\n")

if __name__ == "__main__":
    print("\nRAG bot ready! 🚀")

    while True:
        query = input("\nAsk me something (or type 'exit'): ")
        if query.lower() in ["exit", "quit", "q"]:
            break
        ask(query)
