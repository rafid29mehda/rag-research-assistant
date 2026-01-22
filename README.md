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




## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Research Assistant                    │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
        ┌───────▼────────┐         ┌───────▼────────┐
        │   RETRIEVAL    │         │   GENERATION   │
        │                │         │                │
        │ Stage 1: Hybrid│────────▶│  Prompt Build  │
        │ -  Dense (70%)  │         │                │
        │ -  Sparse (30%) │         │  FLAN-T5 Model │
        │                │         │                │
        │ Stage 2: Rerank│         │  Citation      │
        │ -  Cross-Encoder│         │  Extraction    │
        └────────────────┘         └────────────────┘
                │                           │
        ┌───────▼───────────────────────────▼────────┐
        │      ChromaDB Vector Database               │
        │  -  384-dim embeddings                       │
        │  -  Persistent storage                       │
        │  -  L2 distance similarity                   │
        └─────────────────────────────────────────────┘
```



## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Database** | ChromaDB | Document storage & retrieval |
| **Embeddings** | Sentence-Transformers | Text vectorization |
| **Sparse Retrieval** | BM25 | Keyword-based search |
| **Reranking** | Cross-Encoder | Result optimization |
| **LLM** | FLAN-T5 Large | Answer generation |
| **Framework** | LangChain | Document processing |
| **UI** | Gradio | Web interface |

### Models Used

| Model | Size | Purpose |
|-------|------|---------|
| `all-MiniLM-L6-v2` | 80MB | Document embeddings |
| `ms-marco-MiniLM-L-6-v2` | 90MB | Reranking |
| `google/flan-t5-large` | 3GB | Answer generation |

## Configuration

Key parameters in `src/config.py`:

```python
chunk_size = 500           
chunk_overlap = 50         
top_k_retrieval = 20       
top_k_rerank = 5          
dense_weight = 0.7        
sparse_weight = 0.3       
```





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
```


## Acknowledgments

- **ChromaDB** - Vector database
- **LangChain** - Document processing framework
- **Sentence-Transformers** - Embedding models
- **Hugging Face** - Pre-trained language models
- **Gradio** - Web interface framework
