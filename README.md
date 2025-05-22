
# 📚 NLP Project: Retrieval-Augmented Generation with FAISS & T5


This project implements a Retrieval-Augmented Generation (RAG) pipeline that answers natural language questions using document context. It combines a document retriever (FAISS + SentenceTransformers) with a generator (T5) to generate grounded, context-aware answers.

## 🧠 What is Vector Search?
Vector search allows us to search through large sets of documents by converting text into semantic embeddings—numerical vector representations. Using Sentence-Transformers, we generate these embeddings, which are indexed using FAISS (Facebook AI Similarity Search) for fast and accurate similarity search.

## ⚙️ How the System Works
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

## 🧱 Project Structure

- nlp_project/
  - baseline/
  - data/                 # Input data (.txt, .md, .pdf files)
  - generator/
    - generator.py        # FLAN-T5 answer generation model
  - retriever/
    - retriever.py        # Document chunking, embedding, FAISS retrieval
  - pipeline.py           # End-to-end pipeline script
  - retriever_data/       # Saved FAISS index and documents
  - README.md             # Project documentation 


🚀 Quick Start
✅ Install Requirements

pip install sentence-transformers faiss-cpu transformers torch


### 🗂️ Add Documents
Put your .txt, .md, or .pdf files into the baseline/data/ folder.

Example:

baseline/data/winnie_the_pooh.txt
❓ Ask a Question
Run the pipeline with:

python baseline/pipeline.py
Example Output:

Who is always sad?
➤ Answer: Eeyore is always sad.

🧭 Retriever Module
The Retriever class is designed to efficiently handle document ingestion, preprocessing, chunking, embedding, and similarity search using FAISS.

## 🔨 How It Works:
### Document Chunking
Real-world documents like .txt, .md, or .pdf files can be long and unstructured. To enable efficient retrieval:
The retriever splits large documents into smaller, manageable chunks.
An overlap between consecutive chunks ensures contextual continuity.
This approach improves retrieval performance by preserving semantic meaning across boundaries.

### Text Embedding

Each chunk is passed through a SentenceTransformer model (e.g., all-MiniLM-L6-v2) to generate a dense vector (embedding) that captures its semantic content.
These embeddings are stored in a NumPy array for indexing.

### FAISS Indexing

FAISS (Facebook AI Similarity Search) is used to build a fast similarity search index from the chunk embeddings.
This allows efficient nearest-neighbor search using L2 or cosine similarity, even for large datasets.

### Querying

A user’s query is also converted into an embedding using the same transformer model.
FAISS then retrieves the top-k most semantically similar chunks to serve as context for answer generation.

### Persistence
The retriever supports saving and loading the FAISS index and document metadata for reuse without re-processing.

### 🧪 Retriever Class Overview:
retriever = Retriever()
retriever.add_documents(["myfile.txt"])
retriever.save()  # Save index
retriever.load()  # Load existing index
results = retriever.query("What is FAISS?")



## 🧠 Generator Module

The Generator class uses a text-to-text model (like Flan-T5) to generate fluent and context-aware answers based on retrieved information.

## 🔍 How It Works:
Prompt Construction

A question is combined with the retrieved context chunks to form a structured prompt.

This prompt follows an instruction format like:
Answer the question based on the context.

Context:
<retrieved text from file>

Question:
<user's question>
Answer Generation

The prompt is tokenized and passed into the pretrained Flan-T5 model.
The model generates an answer in natural language using the context.
The output is decoded and returned as the final response.

### ⚙️ Generator Class Overview:
generator = Generator()
context = "\n\n".join(retrieved_chunks)
answer = generator.generate_answer(context, "What is vector search?")

### 📝 Example Usage (in pipeline.py)

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
    
## 💾 Save & Load Retriever State

retriever.save("retriever_data/")
retriever.load("retriever_data/")

## 📬 Future Work
Add support for .pdf and .md parsing
Improve chunking strategy using NLP-based sentence segmentation
Integrate feedback loops for evaluation and improvement
Build a simple UI interface

## 👥 Authors
Team Neurons

https://github.com/khawar-khan520/NLProc-proj-M-SS25/
