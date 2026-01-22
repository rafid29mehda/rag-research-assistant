import gradio as gr
from src.pipeline import RAGPipeline
from src.config import RAGConfig

class GradioInterface:
    def __init__(self, pipeline: RAGPipeline, config: RAGConfig, vector_store):
        self.pipeline = pipeline
        self.config = config
        self.vector_store = vector_store
    
    def answer_question_interface(self, question: str):
        if not question or question.strip() == "":
            return "Please enter a question.", "", "N/A", 0
        
        response = self.pipeline.answer_question(question)
        answer = response['answer']
        confidence = response['confidence']
        num_citations = response['num_citations']
        
        sources_text = ""
        if response['sources']:
            sources_text = "📖 Source Documents:\n" + "-" * 50 + "\n"
            for src in response['sources']:
                cited_marker = "✓ CITED" if src['cited'] else "  (not cited)"
                sources_text += f"[{src['doc_num']}] {src['source']} (Page {src['page']}) - {cited_marker}\n"
                sources_text += f"    Relevance: {src['score']:.4f}\n"
        else:
            sources_text = "No sources available"
        
        return answer, sources_text, confidence, num_citations
    
    def batch_process_interface(self, questions_text: str):
        if not questions_text or questions_text.strip() == "":
            return "Please enter questions (one per line)"
        
        questions = [q.strip() for q in questions_text.split('\n') if q.strip()]
        
        if not questions:
            return "No valid questions found"
        
        responses = self.pipeline.batch_answer(questions)
        
        output = f"Processed {len(responses)} questions\n\n" + "=" * 80 + "\n"
        
        for i, response in enumerate(responses, 1):
            output += f"\nQuestion {i}: {response['question']}\n"
            output += f"Confidence: {response['confidence']}\n"
            output += f"Answer: {response['answer']}\n"
            output += f"Citations: {response['num_citations']}\n"
            output += "-" * 80 + "\n"
        
        return output
    
    def system_info_interface(self):
        return f"""
🖥️ RAG System Information
{'='*60}

Configuration:
• Embedding Model: {self.config.embedding_model}
• Reranker Model: {self.config.reranker_model}
• LLM Model: {self.config.llm_model}
• Device: {self.config.device}

Parameters:
• Chunk Size: {self.config.chunk_size}
• Top-K Retrieval: {self.config.top_k_retrieval}
• Top-K Rerank: {self.config.top_k_rerank}

Vector Store:
• Collection: {self.config.collection_name}
• Documents: {self.vector_store.collection.count()}

{'='*60}
✅ System Ready
        """
    
    def launch(self):
        with gr.Blocks(title="RAG Research Assistant", theme=gr.themes.Soft()) as demo:
            gr.Markdown("# 🤖 RAG Research Assistant\n### AI-powered question answering for ML research")
            
            with gr.Tabs():
                with gr.Tab("💬 Ask Questions"):
                    with gr.Row():
                        with gr.Column():
                            question_input = gr.Textbox(
                                label="Enter your question",
                                placeholder="e.g., What is differential privacy?",
                                lines=3
                            )
                            ask_button = gr.Button("Get Answer", variant="primary")
                            
                            gr.Examples(
                                examples=[
                                    "What is differential privacy?",
                                    "How do attention mechanisms work?",
                                    "Explain federated learning",
                                    "What is SHAP in XAI?",
                                    "How does Byzantine-robust aggregation work?"
                                ],
                                inputs=question_input
                            )
                        
                        with gr.Column():
                            answer_output = gr.Textbox(label="Answer", lines=8, interactive=False)
                            
                            with gr.Row():
                                confidence_output = gr.Textbox(label="Confidence", interactive=False, scale=1)
                                citations_output = gr.Number(label="Citations", interactive=False, scale=1)
                    
                    sources_output = gr.Textbox(label="📚 Sources", lines=6, interactive=False)
                    
                    ask_button.click(
                        fn=self.answer_question_interface,
                        inputs=question_input,
                        outputs=[answer_output, sources_output, confidence_output, citations_output]
                    )
                
                with gr.Tab("📝 Batch Processing"):
                    gr.Markdown("Enter multiple questions (one per line)")
                    batch_input = gr.Textbox(label="Questions", placeholder="What is differential privacy?\nHow do transformers work?", lines=10)
                    batch_button = gr.Button("Process All", variant="primary")
                    batch_output = gr.Textbox(label="Results", lines=20, interactive=False)
                    batch_button.click(fn=self.batch_process_interface, inputs=batch_input, outputs=batch_output)
                
                with gr.Tab("ℹ️ System Info"):
                    info_button = gr.Button("Show System Information", variant="primary")
                    info_output = gr.Textbox(label="Configuration", lines=25, interactive=False)
                    info_button.click(fn=self.system_info_interface, inputs=None, outputs=info_output)
        
        demo.launch(share=True, debug=False, show_error=True)
