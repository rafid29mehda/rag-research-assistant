import os
import re
from pathlib import Path
from typing import List, Dict

def create_project_structure():
    folders = [
        'data/pdfs',
        'data/docs',
        'data/images',
        'data/processed',
        'vectorstore',
        'models',
        'cache'
    ]
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def get_sample_documents() -> List[Dict]:
    return [
        {
            'text': """Attention mechanisms have revolutionized natural language processing.
The transformer architecture, introduced in 'Attention is All You Need' by Vaswani et al.,
uses self-attention to process sequences in parallel. This mechanism allows models to weigh
the importance of different words when encoding or decoding. The key innovation is the
scaled dot-product attention, which computes attention weights using queries, keys, and values.
Multi-head attention enables the model to attend to information from different representation
subspaces simultaneously. This has led to breakthrough models like BERT, GPT, and T5.""",
            'metadata': {'source': 'attention_mechanisms.pdf', 'page': 1, 'type': 'pdf', 'topic': 'NLP'}
        },
        {
            'text': """Federated Learning enables training machine learning models across decentralized devices
while keeping data localized. Introduced by Google in 2016, this paradigm addresses privacy
concerns by never transferring raw data to central servers. Instead, devices train local models
and only share model updates (gradients or weights). The FederatedAveraging algorithm aggregates
these updates to improve the global model. Key challenges include non-IID data distribution,
communication efficiency, and Byzantine-robust aggregation to handle malicious participants.""",
            'metadata': {'source': 'federated_learning.pdf', 'page': 1, 'type': 'pdf', 'topic': 'Machine Learning'}
        },
        {
            'text': """Differential Privacy provides mathematical guarantees for privacy protection in data analysis.
The epsilon-delta framework quantifies privacy loss. A smaller epsilon means stronger privacy
but less accuracy. DP-SGD (Differentially Private Stochastic Gradient Descent) adds calibrated
noise to gradients during training. The privacy budget accumulates over training iterations,
requiring careful management. Recent work shows that adaptive clipping and learning rate schedules
can improve the utility-privacy tradeoff. Applications include private data aggregation and
federated learning with formal privacy guarantees.""",
            'metadata': {'source': 'differential_privacy.pdf', 'page': 1, 'type': 'pdf', 'topic': 'Privacy'}
        },
        {
            'text': """Byzantine-robust aggregation is critical for federated learning security. Byzantine attacks
involve malicious participants sending corrupted updates to poison the global model. Defense
mechanisms include Krum, which selects updates closest to the majority, and Median aggregation,
which computes coordinate-wise medians. Trimmed Mean removes extreme values before averaging.
Recent approaches use clustering to identify and exclude outlier updates. The challenge is
maintaining robustness while not rejecting legitimate updates from non-IID data distributions.""",
            'metadata': {'source': 'byzantine_robust.pdf', 'page': 1, 'type': 'pdf', 'topic': 'Security'}
        },
        {
            'text': """Retrieval-Augmented Generation (RAG) combines retrieval and generation for improved question
answering. The system first retrieves relevant documents from a knowledge base using dense or
sparse retrieval. These documents provide context to a language model for generation. RAG reduces
hallucination by grounding responses in retrieved facts. Hybrid retrieval combining BM25 and
dense embeddings often outperforms single methods. Reranking with cross-encoders improves
relevance. Key challenges include handling multi-hop reasoning and maintaining citation accuracy.""",
            'metadata': {'source': 'rag_systems.pdf', 'page': 1, 'type': 'pdf', 'topic': 'RAG'}
        },
        {
            'text': """XAI (Explainable AI) methods like SHAP and LIME provide interpretations for black-box models.
SHAP (SHapley Additive exPlanations) uses game theory to assign importance scores to features.
LIME (Local Interpretable Model-agnostic Explanations) approximates the model locally with
interpretable models. Attention mechanisms inherently provide some interpretability by showing
which inputs the model focuses on. However, attention weights don't always correlate with
feature importance. GradCAM visualizes which image regions influence predictions in CNNs.
Feature attribution methods help build trust and debug models.""",
            'metadata': {'source': 'xai_methods.pdf', 'page': 1, 'type': 'pdf', 'topic': 'XAI'}
        }
    ]
