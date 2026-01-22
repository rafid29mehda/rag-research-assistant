import re
import torch
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, T5ForConditionalGeneration
from langchain.schema import Document
from src.config import RAGConfig

class PromptTemplate:
    @staticmethod
    def build_rag_prompt(query: str, retrieved_docs: List[Tuple[Document, float]], 
                         max_context_length: int = 800) -> Tuple[str, List[Dict]]:
        context_parts = []
        citation_info = []
        current_length = 0
        
        for i, (doc, score) in enumerate(retrieved_docs):
            doc_num = i + 1
            text = doc.page_content.strip()
            
            if current_length + len(text) > max_context_length:
                remaining = max_context_length - current_length
                if remaining > 100:
                    text = text[:remaining] + "..."
                else:
                    break
            
            context_parts.append(f"[{doc_num}] {text}")
            citation_info.append({
                'doc_num': doc_num,
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A'),
                'topic': doc.metadata.get('topic', 'N/A'),
                'score': score
            })
            
            current_length += len(text)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the context below, answer the question in 2-3 clear sentences. Add [1], [2] citations after facts.

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt, citation_info
    
    @staticmethod
    def extract_citations(text: str) -> List[int]:
        citations = re.findall(r'\[(\d+)\]', text)
        return [int(c) for c in citations]
    
    @staticmethod
    def clean_answer(text: str) -> str:
        text = text.strip()
        
        lines = text.split('\n')
        seen = set()
        unique_lines = []
        
        for line in lines:
            line_clean = re.sub(r'\[\d+\]', '', line).strip().lower()
            if line_clean and line_clean not in seen and len(line_clean) > 10:
                seen.add(line_clean)
                unique_lines.append(line.strip())
        
        text = ' '.join(unique_lines)
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s*\[\s*(\d+)\s*\]\s*', r' [\1] ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

class T5Generator:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model)
        self.model = T5ForConditionalGeneration.from_pretrained(
            config.llm_model,
            device_map=None
        )
        
        self.model = self.model.to(config.device)
        
        if config.device == "cuda":
            self.model = self.model.half()
        
        self.model.eval()
    
    def generate(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                min_length=40,
                temperature=0.8,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                repetition_penalty=2.5,
                length_penalty=1.0,
                no_repeat_ngram_size=4,
                num_beams=1,
                early_stopping=False
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = PromptTemplate.clean_answer(answer)
        
        return answer
    
    def count_tokens(self, text: str) -> int:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)
