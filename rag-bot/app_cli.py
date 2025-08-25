from rag_cli import ask, ingest_new_file, create_qa_chain
import os
import time

def main():
    print("====================================")
    print("   Welcome to the RAG Chat CLI!     ")
    print("====================================")
    print("Type 'upload' to add a PDF or 'exit' to quit.\n")

    qa_chain = None

    while True:
        if qa_chain:
            user_input = input("\nEnter your question (or type 'back' to upload another PDF): ")
        else:
            user_input = input("\nYour input: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        elif user_input.lower() == "upload":
            files = input("Enter PDF file path(s), separated by commas: ").split(",")
            for f in files:
                f = f.strip()
                if os.path.exists(f) and f.endswith(".pdf"):
                    start_time = time.time()
                    retriever = ingest_new_file(f)
                    qa_chain = create_qa_chain(retriever)  # only this PDF
                    end_time = time.time()
                    print(f"⏱ Time taken to ingest {os.path.basename(f)}: {round(end_time - start_time,2)} seconds\n")
                else:
                    print(f"⚠️ File not found or not a PDF: {f}")

        elif user_input.lower() == "back":
            qa_chain = None  # reset to allow new PDF

        elif qa_chain:
            start_time = time.time()
            ask(user_input, qa_chain)
            end_time = time.time()
            print(f"\n⏱ Total latency for this query: {round(end_time - start_time,2)} seconds")

        else:
            print("⚠️ Please upload a PDF first using 'upload'.")

if __name__ == "__main__":
    main()
