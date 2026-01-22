from typing import Dict, List
from src.retrieval import TwoStageRetriever
from src.generation import T5Generator, PromptTemplate
from src.config import RAGConfig

class RAGPipeline:
    def __init__(self, retriever: TwoStageRetriever, llm: T5Generator, config: RAGConfig):
        self.retriever = retriever
        self.llm = llm
        self.config = config
    
    def answer_question(self, question: str, return_sources: bool = True) -> Dict:
        if not question or question.strip() == "":
            return {
                'question': question,
                'answer': "Please enter a question.",
                'confidence': 'EMPTY',
                'top_score': 0.0,
                'sources': [],
                'num_sources': 0,
                'num_citations': 0
            }
        
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
                     'good evening', 'howdy', 'sup', "what's up", 'yo']
        if question.lower().strip() in greetings:
            return {
                'question': question,
                'answer': "Hello! I'm your AI research assistant. Ask me about ML topics like Differential Privacy, Federated Learning, Attention Mechanisms, XAI, or RAG Systems.",
                'confidence': 'GREETING',
                'top_score': 0.0,
                'sources': [],
                'num_sources': 0,
                'num_citations': 0
            }
        
        if len(question.split()) < 3 and '?' not in question:
            return {
                'question': question,
                'answer': f"Query too short. Please ask a more specific question.",
                'confidence': 'TOO_SHORT',
                'top_score': 0.0,
                'sources': [],
                'num_sources': 0,
                'num_citations': 0
            }
        
        retrieved_docs = self.retriever.retrieve(question)
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': "No relevant information found in knowledge base.",
                'confidence': 'NO_RESULTS',
                'top_score': 0.0,
                'sources': [],
                'num_sources': 0,
                'num_citations': 0
            }
        
        retrieved_docs = [(doc, score) for doc, score in retrieved_docs if score > 0.5]
        
        if not retrieved_docs:
            return {
                'question': question,
                'answer': "No relevant information found in knowledge base.",
                'confidence': 'NO_RESULTS',
                'top_score': 0.0,
                'sources': [],
                'num_sources': 0,
                'num_citations': 0
            }
        
        top_score = retrieved_docs[0][1]
        
        if top_score >= 8.0:
            confidence = "HIGH"
        elif top_score >= 5.0:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        if confidence == "LOW" and top_score < 3.0:
            return {
                'question': question,
                'answer': "Not enough relevant information. Try asking about Differential Privacy, Federated Learning, Attention Mechanisms, XAI, or RAG.",
                'confidence': 'OUT_OF_SCOPE',
                'top_score': top_score,
                'sources': [],
                'num_sources': 0,
                'num_citations': 0
            }
        
        prompt, citation_info = PromptTemplate.build_rag_prompt(question, retrieved_docs, max_context_length=900)
        answer = self.llm.generate(prompt, max_length=200, temperature=0.7)
        cited_docs = PromptTemplate.extract_citations(answer)
        
        sources = []
        for cite in citation_info:
            sources.append({
                'doc_num': cite['doc_num'],
                'source': cite['source'],
                'page': cite['page'],
                'score': cite['score'],
                'cited': cite['doc_num'] in cited_docs
            })
        
        return {
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'top_score': top_score,
            'sources': sources if return_sources else [],
            'num_sources': len(sources),
            'num_citations': len(cited_docs)
        }
    
    def batch_answer(self, questions: List[str]) -> List[Dict]:
        return [self.answer_question(question) for question in questions]
