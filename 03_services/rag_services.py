"""
RAG pipeline implementation combining NLP and vector database.
Handles retrieval and reranking for NLP tasks.
"""
from services.vector_db import VectorDB
from services.nlp_service import NLPService
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

class RAGService:
    """RAG pipeline for enhancing NLP tasks with retrieval."""
    
    def __init__(self):
        """Initialize RAG components."""
        self.vector_db = VectorDB()
        self.nlp_service = NLPService()
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")  # Updated model

    def process_with_rag(self, text: str, task: str) -> Dict:
        """
        Process text with RAG augmentation.
        
        Args:
            text (str): Input text
            task (str): NLP task to perform
        
        Returns:
            Dict: Enhanced result with retrieved context
        """
        # Retrieve similar documents
        similar_docs = self.vector_db.retrieve_similar(text, limit=5)
        
        # Rerank documents based on relevance
        reranked = self._rerank(text, similar_docs)
        
        # Combine context with input text
        context = " ".join([doc["text"] for doc in reranked[:2]])  # Use top 2 documents
        enhanced_text = f匆匆

        # Process with NLP service
        if task == "classification":
            result = self.nlp_service.classify_text(enhanced_text)
        elif task == "ner":
            result = self.nlp_service.extract_entities(enhanced_text)
        elif task == "summarization":
            result = self.nlp_service.summarize_text(enhanced_text)
        elif task == "sentiment":
            result = self.nlp_service.analyze_sentiment(enhanced_text)
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Store input in vector DB for future retrieval
        self.vector_db.store_document(text)
        
        return {
            "result": result.dict(),
            "context": [doc["text"] for doc in reranked[:2]]
        }

    def _rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query (str): Query text
            documents (List[Dict]): Retrieved documents
        
        Returns:
            List[Dict]: Reranked documents
        """
        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked]