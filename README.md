# nlp_project

Vector Search with FAISS and T5

What is Vector Search?
Vector search allows us to search through a large set of documents or text chunks using embeddings—numerical vector representations of the text. Each text chunk is converted into a vector using a model like Sentence-Transformers. FAISS (Facebook AI Similarity Search) is then used to perform fast similarity search, identifying the most relevant chunks based on a given query.

How Does It Work?
Text Embedding:
The text (e.g., chapters, paragraphs) is split into smaller, manageable chunks.

Each chunk is transformed into a fixed-length embedding vector using the SentenceTransformer model. These vectors capture the semantic meaning of the text.

FAISS Indexing:

The embeddings are indexed using FAISS. FAISS is an open-source library designed for efficient similarity search and clustering of dense vectors.

The indexing process allows FAISS to quickly retrieve the nearest neighbor embeddings based on cosine similarity.

Querying:

When you enter a query, it is also transformed into an embedding vector using the same model.

FAISS performs a similarity search to find the top-k most relevant text chunks by comparing the cosine similarity between the query embedding and the pre-stored chunk embeddings.

Answer Generation:

The top-k matching text chunks are then used as context for the T5 model, which generates an answer to the query based on the retrieved chunks.

Why Use Vector Search?
Vector search makes it possible to efficiently search through large datasets without having to rely on keyword-based searches, which can be less effective in understanding the true meaning of the text. By using embeddings, you ensure that the search captures the semantic similarity between the query and the text, making the system more effective at answering a wide range of questions.
