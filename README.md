# RAG Research Assistant

Advanced Retrieval-Augmented Generation system for answering questions about machine learning research topics using a two-stage retrieval pipeline with vector database storage.

## Features

- **Two-Stage Retrieval**: Hybrid search (BM25 + Dense Embeddings) + Cross-Encoder reranking
- **Vector Database**: Persistent ChromaDB for efficient similarity search
- **Citation Management**: Automatic source citation with relevance scoring
- **Off-Topic Detection**: Intelligent handling of out-of-scope queries
- **Interactive UI**: Gradio web interface

## Topics Covered

- Differential Privacy & DP-SGD
- Federated Learning
- Attention Mechanisms & Transformers
- Explainable AI (XAI)
- Byzantine-Robust Aggregation
- RAG Systems


## Setup

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/rag-research-assistant.git
cd rag-research-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Setup vector store
python setup_data.py

# Run application
python main.py
