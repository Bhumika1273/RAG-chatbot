# app_cli.py

from rag import ask, ingest_new_file
import os
import time

def main():
    print("====================================")
    print("   Welcome to the RAG Chat CLI!     ")
    print("====================================")
    print("Type your question about the document, 'upload' to add PDFs, or 'exit' to quit.\n")

    while True:
        user_input = input("Your input: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break

        elif user_input.lower() == "upload":
            files = input("Enter PDF file path(s), separated by commas: ").split(",")
            for f in files:
                f = f.strip()
                if os.path.exists(f) and f.endswith(".pdf"):
                    start_time = time.time()
                    ingest_new_file(f)
                    end_time = time.time()
                    print(f"⏱ Time taken to ingest {os.path.basename(f)}: {round(end_time - start_time,2)} seconds\n")
                else:
                    print(f"⚠️ File not found or not a PDF: {f}")

        else:
            start_time = time.time()
            ask(user_input)
            end_time = time.time()
            print(f"\n⏱ Total latency for this query: {round(end_time - start_time,2)} seconds")
            print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
