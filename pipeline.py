import os
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
from baseline.prepare_data.prepare_data import load_and_prepare_medquad

class MedicalAssistant:
    def __init__(self):
        self.index_path = "baseline/retriever/faiss_index.pkl"
        self.data_path = "baseline/data/medquad.csv"
        self.retriever = Retriever()

        if os.path.exists(self.index_path):
            print("[MedicalAssistant] Loading existing FAISS index...")
            self.retriever.load_index(self.index_path)
        else:
            print("[MedicalAssistant] Building FAISS index from MedQuAD...")
            docs = load_and_prepare_medquad(self.data_path)
            self.retriever.build_index(docs)
            self.retriever.save_index(self.index_path)

        self.generator = Generator()

    def answer(self, question, top_k=3):
        # Retrieve relevant chunks from retriever
        context_chunks = self.retriever.retrieve(question, top_k=top_k)

        if not context_chunks:
            return context_chunks, "Sorry, I couldn’t find relevant medical information."

        # Join chunks as context input for generator
        context = " ".join(context_chunks)

        # Generate answer using your generator model
        generated_answer = self.generator.generate(question, context)

        return context_chunks, generated_answer


def main():
    # Your original interactive loop, unchanged:
    retriever = Retriever()
    index_path = "baseline/retriever/faiss_index.pkl"
    data_path = "baseline/data/medquad.csv"

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

        if not context_chunks:
            print("🤖 Assistant: Sorry, I couldn’t find relevant medical information. Please ask a medical question.")
            print("-" * 60)
            continue

        print("\n📚 Top 3 Relevant Chunks:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"\n🔹 Chunk {i}:\n{chunk.strip()}")

        answer = generator.generate(user_question, context_chunks)

        print(f"\n🤖 Assistant: {answer}\n{'-'*60}")

if __name__ == "__main__":
    main()
