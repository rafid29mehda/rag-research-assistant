from src.config import RAGConfig
from src.models import EmbeddingGenerator
from src.retrieval import VectorStore, BM25SparseRetriever, HybridRetriever, CrossEncoderReranker, TwoStageRetriever
from src.generation import T5Generator
from src.pipeline import RAGPipeline
from src.interface import GradioInterface
from setup_data import setup_vector_store

def main():
    vector_store, chunked_documents, embedding_generator, config = setup_vector_store()
    
    bm25_retriever = BM25SparseRetriever(chunked_documents)
    
    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        bm25_retriever=bm25_retriever,
        dense_weight=config.dense_weight,
        sparse_weight=config.sparse_weight
    )
    
    reranker = CrossEncoderReranker(config)
    
    two_stage_retriever = TwoStageRetriever(
        hybrid_retriever=hybrid_retriever,
        reranker=reranker,
        config=config
    )
    
    llm_generator = T5Generator(config)
    
    rag_pipeline = RAGPipeline(
        retriever=two_stage_retriever,
        llm=llm_generator,
        config=config
    )
    
    interface = GradioInterface(rag_pipeline, config, vector_store)
    interface.launch()

if __name__ == "__main__":
    main()
