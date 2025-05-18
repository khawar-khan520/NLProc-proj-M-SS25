
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

class Retriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None

    def chunk_text(self, text, chunk_size=200, overlap=50):
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            chunks.append(chunk)
        return chunks

    def add_documents(self, texts):
        chunks = []
        for text in texts:
            chunks.extend(self.chunk_text(text))
        self.documents.extend(chunks)
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        self.embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def query(self, question, top_k=3):
        query_embedding = self.model.encode([question]).astype("float32")
        D, I = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in I[0]]

    def save(self, path="retriever_data"):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path="retriever_data"):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
