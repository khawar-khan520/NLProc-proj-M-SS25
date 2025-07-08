🤖 Medical QA System using Retrieval-Augmented Generation (RAG)

This project implements a powerful RAG (Retrieval-Augmented Generation) pipeline for Medical Question Answering using state-of-the-art NLP models. It allows users to ask questions and get context-rich answers grounded in real medical data from MedQuAD.

🧠 What is This Project About?

This system acts as a Medical Assistant, answering health-related questions using evidence from trusted medical documents. It combines:

🔍 Document retrieval using FAISS + Sentence Transformers

🧾 Context-aware generation using FLAN-T5

📊 Evaluation tools using F1 score for accuracy checking

Whether you're researching or building healthcare AI, this tool gives you medically grounded answers and supports interactive or batch evaluation.

⚙️ How It Works

🔹 1. Chunking Documents

Long documents are split into overlapping chunks (e.g., 200 tokens with 50-token overlap) to preserve meaning and improve retrieval quality.

🔹 2. Embedding with SentenceTransformers

Each chunk is converted into a semantic vector using the all-MiniLM-L6-v2 model.

🔹 3. FAISS Indexing

The dense vectors are stored in a FAISS index for ultra-fast nearest neighbor search based on cosine similarity.

🔹 4. Question Answering

User questions are embedded, matched to relevant chunks, and passed to FLAN-T5 to generate the final answer.

🧪 Evaluation Modes

✅ Interactive Evaluation

Run interactive_eval.py

Ask questions in the terminal

See:

Top 3 retrieved chunks 📚

Answer from the model 🤖

F1 score based on your question input 📊

python interactive_eval.py

✅ Batch Evaluation

Run eval.py with a .json file of Q&A pairs

Evaluates answers and computes F1 scores automatically

python eval.py --input baseline/data/test_inputs.json

Example test file:

[
  {"question": "What is diabetes?", "answer": "Diabetes is a chronic condition..."},
  {"question": "How to prevent heart attack?", "answer": "Follow a heart-healthy lifestyle..."}
]

🧱 Project Structure

NLProc-proj-M-SS25/
├── baseline/
│   ├── data/                # Input data (.csv or .txt)
│   ├── retriever/           # FAISS-based retriever
│   ├── generator/           # FLAN-T5 based generator
│   ├── retriever_data/      # Saved FAISS index and doc chunks
│   └── medquad.csv          # Medical Q&A data
├── pipeline.py              # Main RAG pipeline logic
├── interactive_eval.py      # Interactive terminal QA tool
├── eval.py                  # Batch evaluation script (JSON-based)
└── README.md

📦 Installation

pip install -r requirements.txt

Or individually:

pip install sentence-transformers faiss-cpu transformers torch

💡 Example

Enter question: what is prevention for heart attack?

📚 Top 3 Chunks:
🔹 Chunk 1: Heart-Healthy Lifestyle (diet, exercise, manage diabetes...)
🔹 Chunk 2: Risk factor management (smoking, weight, emergency plan...)
🔹 Chunk 3: Cholesterol control, blood pressure treatment...

🤖 Answer: Talk to your doctor about the signs of a heart attack.
📊 F1 score (vs. input): 0.353

🧑‍💻 Authors

Team Neurons – Master’s Project @ University of Bamberg

GitHub: khawar-khan520/NLProc-proj-M-SS25

🚀 Future Enhancements

Add UI for live demo

PDF/Markdown support

Improve chunking with NLP sentence segmentation

Add multilingual support

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------Enjoy exploring the intersection of AI + Medicine! 🩺📊🤖
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 👥 Authors
Team Neurons

https://github.com/khawar-khan520/NLProc-proj-M-SS25/
