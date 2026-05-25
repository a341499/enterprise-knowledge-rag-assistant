\# Enterprise Knowledge RAG Assistant



A retrieval-augmented generation assistant for answering questions from enterprise engineering knowledge documents.



## Architecture Diagram



![Enterprise Knowledge RAG Architecture](architecture/enterprise\_knowledge\_rag\_architecture.png)



\## Problem



Enterprise engineering teams often depend on scattered documents, release notes, design notes, regression reports, and debugging summaries. Searching manually is slow and context is easily lost.



\## Solution



This project builds a RAG-based assistant that retrieves relevant document chunks and generates grounded answers with source references.



\## Architecture



Documents → Chunking → Embeddings → Vector Database → Retriever → LLM → Answer with citations



\## Key Features



\- Document ingestion

\- Text chunking

\- Embedding generation

\- Vector search

\- Context-aware answer generation

\- Source citation support

\- Streamlit user interface



\## Tech Stack



\- Python

\- Streamlit

\- ChromaDB

\- Sentence Transformers

\- LangChain / LlamaIndex

\- OpenAI-compatible LLM API



\## Portfolio Relevance



This project demonstrates practical AI infrastructure skills:

\- RAG system design

\- Vector database usage

\- Retrieval quality evaluation

\- Enterprise knowledge assistant architecture

\- LLM application engineering

