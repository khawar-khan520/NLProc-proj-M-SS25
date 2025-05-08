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


---

##  Retriever Module

We implemented a reusable `Retriever` class that supports:

- Chunking and indexing text documents
- Querying using natural language
- Saving and loading with FAISS

In this second task, we aim to:

Handle real documents like .txt, .md, or .pdf files.

Preprocess these documents and split them into smaller chunks.

Index the chunks using FAISS.

Use FAISS to retrieve the most relevant document chunks for a given query.

How Does It Work?
1. Document Chunking:
Real documents, such as text files, markdown files, or PDFs, can be large and complex. To process them efficiently, they need to be split into manageable chunks. These chunks are typically smaller sections of text, such as sentences, paragraphs, or logical segments, that make sense on their own.

The retriever uses a chunking function that splits large documents into smaller pieces while ensuring an overlap between consecutive chunks to improve retrieval accuracy.

2. Text Embedding:
Once the documents are chunked, each chunk is transformed into a vector embedding. This is done using the Sentence-Transformers library, which converts text into fixed-length vectors that capture the semantic meaning of the text. These embeddings are stored and indexed for fast retrieval.

3. FAISS Indexing:
The FAISS library (Facebook AI Similarity Search) is used to index the embeddings of the document chunks. FAISS enables efficient similarity search by organizing the embeddings in a way that allows for quick retrieval of the most relevant chunks for a given query.

FAISS works by calculating the cosine similarity between the query vector and the indexed embeddings, identifying the most similar chunks quickly.

4. Querying:
When a user enters a query, the query is also converted into a vector embedding. The retriever searches the FAISS index for the top-k most relevant chunks by comparing the query embedding with the pre-indexed document embeddings.

5. Answer Generation:
Once the top-k relevant chunks are retrieved, these chunks are used as context for generating an answer. In the final implementation, the T5 model (Text-to-Text Transfer Transformer) can be used to generate an answer based on the context provided by the top-k matching chunks.

Key Concepts
Vector Search:
Vector search involves converting text into numerical vector representations, known as embeddings. These embeddings capture the meaning of the text, allowing for searches based on semantic similarity rather than exact word matches. This method is more effective for understanding the meaning behind a query and providing relevant results, especially in large datasets.

Chunking:
Text is divided into smaller chunks, which are easier to process and index. This process ensures that each chunk is small enough to be accurately embedded and indexed.

FAISS:
FAISS is a library developed by Facebook for efficient similarity search. It allows for the rapid retrieval of the nearest neighbors of a given vector, making it ideal for tasks that involve searching through large numbers of document embeddings.

T5 Model:
The T5 (Text-to-Text Transfer Transformer) model is a versatile language model that can perform various NLP tasks, including answering questions. In this context, it is used to generate answers based on the relevant chunks retrieved by FAISS.

Steps for Implementing Task 2
1. Feed Real Documents:
The retriever should be able to process and handle real documents such as .txt, .md, or .pdf files.

Text Preprocessing: The text needs to be cleaned, removing unnecessary characters, and formatted correctly for chunking.

2. Chunking Function:
After preprocessing, the document is split into smaller chunks. This is done using a chunking function that takes into account the chunk size and possible overlap between consecutive chunks to ensure the context is preserved.

3. Indexing:
The chunks are then embedded using Sentence-Transformers.

The embeddings are indexed using FAISS, allowing for efficient retrieval of the most relevant chunks based on cosine similarity.

4. Querying and Retrieval:
When a query is entered, it is transformed into a vector embedding.

FAISS is used to search for the top-k most relevant chunks based on the query embedding and the pre-stored document embeddings.

5. Save and Load:
The FAISS index and documents can be saved to disk and loaded later, ensuring that the state of the retriever is preserved between sessions. This can be done using methods like save() and load().

6. Testing:
The retriever should be tested to ensure it returns the correct chunks based on given queries. This involves running a simple test with a document and a query to verify the expected results.

Methods of the Retriever Class
add_documents(texts):

This method takes in a list of documents (texts), chunks them into smaller parts, and indexes them using FAISS.

query(question, top_k=3):

Given a question, this method retrieves the top-k most relevant chunks from the indexed documents by comparing cosine similarity.

save(path="retriever_data"):

Saves the FAISS index and documents to a specified directory. This is useful for persisting the retriever state for later use.

load(path="retriever_data"):

Loads the FAISS index and documents from the saved directory, resuming the state of the retriever.

Deliverables
1. A retriever.py module with reusable class:
The Retriever class handles document chunking, embedding, indexing, and querying.

2. One or more loaded document sources:
This includes the documents to be processed and indexed.

3. Working local search using queries:
The retriever should allow users to search documents using queries and retrieve the most relevant chunks.

4. Committed README update and usage instructions:
The updated README file provides clear instructions on how to use the retriever, how to add documents, and how to query for answers.

