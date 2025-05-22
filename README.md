
📚 NLP Project: Retrieval-Augmented Generation with FAISS & T5


This project implements a Retrieval-Augmented Generation (RAG) pipeline that answers natural language questions using document context. It combines a document retriever (FAISS + SentenceTransformers) with a generator (T5) to generate grounded, context-aware answers.

🧠 What is Vector Search?
Vector search allows us to search through large sets of documents by converting text into semantic embeddings—numerical vector representations. Using Sentence-Transformers, we generate these embeddings, which are indexed using FAISS (Facebook AI Similarity Search) for fast and accurate similarity search.

⚙️ How the System Works
🔹 1. Document Chunking
Long documents are split into overlapping chunks (e.g., 200 tokens with 50-token overlap) to preserve semantic flow.

🔹 2. Text Embedding
Each chunk is converted into a fixed-length vector using the all-MiniLM-L6-v2 model from Sentence-Transformers.

🔹 3. FAISS Indexing
The embeddings are indexed with FAISS for fast nearest-neighbor search based on cosine similarity.

🔹 4. Querying
When a question is asked, it is embedded and compared against the indexed vectors to retrieve the most relevant text chunks.

🔹 5. Answer Generation
Top-k chunks are provided as context to the FLAN-T5 model, which generates a natural language answer.

🧱 Project Structure

nlp_project/
│
├── baseline/
│   ├── data/                       # Input data (e.g., .txt, .md, .pdf files)
│   ├── generator/
│   │   └── generator.py            # FLAN-T5 model for answer generation
│   ├── retriever/
│   │   └── retriever.py            # Document chunking, embedding, and FAISS retrieval
│   └── pipeline.py                 # End-to-end pipeline script
└── README.md                      # Project documentation (this file)
🚀 Quick Start
✅ Install Requirements

pip install sentence-transformers faiss-cpu transformers torch


🗂️ Add Documents
Put your .txt, .md, or .pdf files into the baseline/data/ folder.

Example:

baseline/data/winnie_the_pooh.txt
❓ Ask a Question
Run the pipeline with:

python baseline/pipeline.py
Example Output:

Who is always sad?
➤ Answer: Eeyore is always sad.

🔍 Retriever Module
The Retriever class handles:

Text chunking with overlap

Embedding via SentenceTransformers

FAISS-based vector indexing and search

Save/load functionality

Key Methods
add_documents(texts): Chunks and indexes input texts

query(question, top_k=3): Returns top-k relevant text chunks

save(path): Persists index and document state

load(path): Loads previously saved state

🧠 Generator Module
The Generator class uses the google/flan-t5-base model to:

Construct prompts with context and question

Generate fluent, context-aware answers

📝 Example Usage (in pipeline.py)

from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator

def run_pipeline(document_path, question):
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()

    retriever = Retriever()
    retriever.add_documents([text])
    results = retriever.query(question)

    context = "\n\n".join(results)
    generator = Generator()
    answer = generator.generate_answer(context, question)

    return answer
💾 Save & Load Retriever State

retriever.save("retriever_data/")
retriever.load("retriever_data/")

📬 Future Work
Add support for .pdf and .md parsing

Improve chunking strategy using NLP-based sentence segmentation

Integrate feedback loops for evaluation and improvement

Build a simple UI interface

👥 Authors
Team Neurons

https://github.com/khawar-khan520/NLProc-proj-M-SS25/
