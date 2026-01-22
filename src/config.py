import os
import warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
warnings.filterwarnings('ignore')

from dataclasses import dataclass
import torch
from pathlib import Path

@dataclass
class RAGConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "google/flan-t5-large"
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    
    collection_name: str = "research_docs"
    persist_directory: str = "./vectorstore"
    
    max_new_tokens: int = 512
    temperature: float = 0.7
    
    @property
    def device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def __post_init__(self):
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
