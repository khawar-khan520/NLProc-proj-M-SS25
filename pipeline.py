import os
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
from baseline.prepare_data.prepare_data import load_and_prepare_medquad
import json

def main():
    data_path = "baseline/data/medquad.csv"
    test_input_path = "baseline/data/test_inputs.json"
    index_path = "baseline/retriever/faiss_index.pkl"

    retriever = Retriever()

    if os.path.exists(index_path):
        print("[Pipeline] Loading existing FAISS index...")
        retriever.load_index(index_path)
    else:
        print("[Pipeline] Building FAISS index from MedQuAD...")
        docs = load_and_prepare_medquad(data_path)
        retriever.build_index(docs)
        retriever.save_index(index_path)

    generator = Generator()

    print("🩺 Welcome to the Medical Assistant!")
    print("Type your medical question (or type 'exit' to quit).\n")

    while True:
        user_question = input("🧠 You: ")

        if user_question.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Stay healthy.")
            break

        context_chunks = retriever.retrieve(user_question, top_k=3)

        # If nothing found
        if not context_chunks:
            print("🤖 Assistant: Sorry, I couldn’t find relevant medical information. Please ask a medical question.")
            print("-" * 60)
            continue

        print("\n📚 Top 3 Relevant Chunks:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"\n🔹 Chunk {i}:\n{chunk.strip()}")

        combined_context = " ".join(context_chunks)
        answer = generator.generate(user_question, combined_context)

        print(f"\n🧠 Final Answer: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    main()
