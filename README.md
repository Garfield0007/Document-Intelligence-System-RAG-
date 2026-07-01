# Document Intelligence System (RAG)

A Retrieval-Augmented Generation (RAG) application that enables users to upload multiple PDF documents and interact with them through natural language queries. The system combines semantic search, vector embeddings, and large language models to deliver accurate, context-aware answers with source attribution.

## Features

* Multi-PDF document ingestion
* Automated text extraction and chunking
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Source-aware responses
* Persistent vector storage for faster repeated queries
* Interactive Streamlit web interface

## How It Works

1. Upload one or more PDF documents.
2. Documents are processed and split into semantic chunks.
3. Embeddings are generated and stored in ChromaDB.
4. User queries are converted into embeddings.
5. Relevant document chunks are retrieved.
6. Llama 3.1 generates a context-aware response using the retrieved information.

## Tech Stack

* Python
* Streamlit
* LangChain
* ChromaDB
* Hugging Face Embeddings (BAAI BGE)
* Groq API
* Llama 3.1

## Key Highlights

* Built an end-to-end RAG pipeline for document question answering.
* Implemented semantic retrieval with source attribution.
* Integrated ChromaDB for efficient vector storage and retrieval.
* Enabled persistent indexing to avoid reprocessing documents on every session.

## Use Cases

* Research document analysis
* Academic paper exploration
* Technical documentation search
* Knowledge base question answering

## Demo

Live Demo: [Add Streamlit Link]

## Screenshots

(Add screenshots here)

## Future Improvements

* Support for DOCX and TXT files
* Hybrid search (keyword + semantic retrieval)
* Conversation memory
* Multi-user authentication
* Advanced evaluation metrics

## Author

Kanishka Chouhan
