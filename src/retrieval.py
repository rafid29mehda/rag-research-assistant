import re
import chromadb
import numpy as np
from typing import List, Tuple, Dict
from langchain.schema import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from src.config import RAGConfig
from src.models import EmbeddingGenerator

class VectorStore:
    def __init__(self, config: RAGConfig, embedding_generator: EmbeddingGenerator):
        self.config = config
        self.embedding_generator = embedding_generator
        self.client = chromadb.PersistentClient(path=config.persist_directory)
        
        try:
            self.collection = self.client.get_collection(name=config.collection_name)
        except:
            self.collection = self.client.create_collection(
                name=config.collection_name,
                metadata={"description": "Research documents for RAG system"}
            )
    
    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = self.embedding_generator.encode_texts(texts)
        
        ids = []
        for i, doc in enumerate(documents):
            doc_id = f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('page', 0)}_{doc.metadata.get('chunk_id', i)}"
            doc_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id)
            ids.append(doc_id)
        
        existing_ids = set()
        try:
            existing = self.collection.get(ids=ids)
            if existing and 'ids' in existing:
                existing_ids = set(existing['ids'])
        except:
            pass
        
        new_texts = []
        new_embeddings = []
        new_metadatas = []
        new_ids = []
        
        for i, doc_id in enumerate(ids):
            if doc_id not in existing_ids:
                new_texts.append(texts[i])
                new_embeddings.append(embeddings[i])
                new_metadatas.append(metadatas[i])
                new_ids.append(doc_id)
        
        if new_ids:
            self.collection.add(
                documents=new_texts,
                embeddings=[emb.tolist() for emb in new_embeddings],
                metadatas=new_metadatas,
                ids=new_ids
            )
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        query_embedding = self.embedding_generator.encode_single(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents_with_scores = []
        for i in range(len(results['documents'][0])):
            doc = Document(
                page_content=results['documents'][0][i],
                metadata=results['metadatas'][0][i]
            )
            distance = results['distances'][0][i]
            similarity = 1 / (1 + distance)
            documents_with_scores.append((doc, similarity))
        
        return documents_with_scores

class BM25SparseRetriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.retriever = BM25Retriever.from_documents(documents)
        self.retriever.k = 20
    
    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        self.retriever.k = k
        results = self.retriever.get_relevant_documents(query)
        
        documents_with_scores = []
        for i, doc in enumerate(results):
            score = max(1.0 - (i * 0.05), 0.1)
            documents_with_scores.append((doc, score))
        
        return documents_with_scores

class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_retriever: BM25SparseRetriever, 
                 dense_weight: float = 0.7, sparse_weight: float = 0.3):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[Document, float, Dict]]:
        dense_results = self.vector_store.similarity_search(query, k=20)
        sparse_results = self.bm25_retriever.search(query, k=20)
        
        dense_scores = {self._get_doc_id(doc): score for doc, score in dense_results}
        sparse_scores = {self._get_doc_id(doc): score for doc, score in sparse_results}
        
        combined_scores = {}
        all_doc_ids = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        for doc_id in all_doc_ids:
            d_score = dense_scores.get(doc_id, 0.0)
            s_score = sparse_scores.get(doc_id, 0.0)
            combined = (self.dense_weight * d_score) + (self.sparse_weight * s_score)
            
            combined_scores[doc_id] = {
                'combined': combined,
                'dense': d_score,
                'sparse': s_score
            }
        
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1]['combined'], reverse=True)[:k]
        
        doc_map = {}
        for doc, _ in dense_results + sparse_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
        
        results = []
        for doc_id, scores in sorted_docs:
            if doc_id in doc_map:
                results.append((doc_map[doc_id], scores['combined'], scores))
        
        return results
    
    def _get_doc_id(self, doc: Document) -> str:
        source = doc.metadata.get('source', 'unknown')
        page = doc.metadata.get('page', 0)
        chunk = doc.metadata.get('chunk_id', 0)
        return f"{source}_{page}_{chunk}"

class CrossEncoderReranker:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = CrossEncoder(config.reranker_model, max_length=512, device=config.device)
    
    def rerank(self, query: str, documents: List[Tuple[Document, float]], 
               top_k: int = 5) -> List[Tuple[Document, float]]:
        if not documents:
            return []
        
        pairs = [[query, doc.page_content] for doc, _ in documents]
        scores = self.model.predict(pairs)
        
        scores = np.array(scores)
        min_score = scores.min()
        if min_score < 0:
            scores = scores - min_score
        
        max_score = scores.max()
        if max_score > 0:
            scores = (scores / max_score) * 10
        
        reranked = [(doc, float(scores[i])) for i, (doc, _) in enumerate(documents)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return reranked[:top_k]

class TwoStageRetriever:
    def __init__(self, hybrid_retriever: HybridRetriever, reranker: CrossEncoderReranker, config: RAGConfig):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.config = config
    
    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        hybrid_results = self.hybrid_retriever.hybrid_search(query, k=self.config.top_k_retrieval)
        initial_docs = [(doc, score) for doc, score, _ in hybrid_results]
        final_results = self.reranker.rerank(query, initial_docs, top_k=self.config.top_k_rerank)
        return final_results
