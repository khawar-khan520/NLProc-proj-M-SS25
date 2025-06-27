from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import faiss
import numpy as np
import pickle
import os

class Retriever:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.docs = []

    def build_index(self, docs):
        self.docs = docs
        embeddings = self.model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
        embeddings = normalize(embeddings)  # Normalize for cosine similarity

        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine
        self.index.add(embeddings)

        print(f"[Retriever] FAISS index built with {len(docs)} documents.")

    def embed_text(self, text):
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding / np.linalg.norm(embedding)  # Normalize query embedding

    def retrieve(self, query, top_k=3, similarity_threshold=0.60):
        query_embedding = self.embed_text(query)  # Use normalized embedding
        D, I = self.index.search(np.array([query_embedding]), top_k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or score < similarity_threshold:
                continue
            results.append(self.docs[idx])

        return results  # May return an empty list

    def save_index(self, path='baseline/retriever/faiss_index.pkl'):
        with open(path, 'wb') as f:
            pickle.dump((self.index, self.docs), f)

    def load_index(self, path='baseline/retriever/faiss_index.pkl'):
        with open(path, 'rb') as f:
            self.index, self.docs = pickle.load(f)
