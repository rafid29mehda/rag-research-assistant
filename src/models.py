import os
import re
from typing import List, Dict, Tuple
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import numpy as np
from src.config import RAGConfig

class PDFProcessor:
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        try:
            reader = PdfReader(pdf_path)
            filename = os.path.basename(pdf_path)
            documents = []
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) < 50:
                    continue
                
                documents.append({
                    'text': text,
                    'metadata': {
                        'source': filename,
                        'page': page_num + 1,
                        'type': 'pdf',
                        'total_pages': len(reader.pages)
                    }
                })
            
            return documents
        except Exception as e:
            return []
    
    def process_directory(self, directory: str) -> List[Dict]:
        all_documents = []
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory, pdf_file)
            docs = self.extract_text_from_pdf(pdf_path)
            all_documents.extend(docs)
        
        return all_documents

class DocumentChunker:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Dict]) -> List[Document]:
        chunked_docs = []
        
        for doc in documents:
            text = doc['text']
            metadata = doc['metadata']
            chunks = self.text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['total_chunks'] = len(chunks)
                
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
        
        return chunked_docs

class EmbeddingGenerator:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model, device=config.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def encode_texts(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=32,
            convert_to_numpy=True
        )
    
    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.encode_single(text1)
        emb2 = self.encode_single(text2)
        return float(np.dot(emb1, emb2))
