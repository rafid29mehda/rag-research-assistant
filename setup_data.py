from src.config import RAGConfig
from src.models import DocumentChunker, EmbeddingGenerator
from src.retrieval import VectorStore
from src.utils import create_project_structure, get_sample_documents

def setup_vector_store():
    create_project_structure()
    
    config = RAGConfig()
    
    sample_documents = get_sample_documents()
    
    chunker = DocumentChunker(config)
    chunked_documents = chunker.chunk_documents(sample_documents)
    
    embedding_generator = EmbeddingGenerator(config)
    vector_store = VectorStore(config, embedding_generator)
    vector_store.add_documents(chunked_documents)
    
    return vector_store, chunked_documents, embedding_generator, config

if __name__ == "__main__":
    setup_vector_store()
