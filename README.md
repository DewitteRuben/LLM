# Overview

This repository is designed to explore and demonstrate the integration of LLMs with vector search technologies. This project showcases various approaches to managing and querying embeddings with the help of Retrieval-Augmented Generation (using FAISS and Chroma).

## Project Structure

The repository is organized into the following directories:

Here’s a follow-up on each point without repetition:

### **Introduction**

An overview of LLMs with examples of building a simple LLM app using LangChain and Python scripts:

- **rag-chain.py**: Focuses on a straightforward RAG pipeline, ideal for learning the basics of embedding storage and context-based querying.  
- **rag-graph.py**: Expands on the pipeline by introducing state graphs, improving the organization and flow of data for more complex use cases.  
- **rag-graph-tools.py**: Takes the concept further by integrating tools/


### **FAISS**
- Centers around [FAISS](https://faiss.ai/), a popular library for efficient similarity search and clustering of dense vectors.
- Includes:
  - **l2-index.py**:  Example script that uses FAISS for vector searching with an L2 index (Euclidean distance).
  - **ivf-index.py**: Example script that uses FAISS for vector searching with an L2 index combined with IVF partitioning.
  - **langchain/**: 
    - **langchain-faiss.py**: Demonstrates FAISS integration with LangChain.

### **Chroma**
- Focuses on using [Chroma](https://docs.trychroma.com/) for vector database management.