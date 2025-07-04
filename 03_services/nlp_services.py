# ../03_services/nlp_services.py

"""
NLP processing service using Hugging Face transformers.
Handles classification, entity extraction, summarization, and sentiment analysis.
"""
from transformers import pipeline
from typing import List, Dict, Any
from config.config import config
from models.schema import (
    ClassificationResponse, EntityResponse,
    SummarizationResponse, SentimentResponse
)

class NLPService:
    """Service for performing NLP tasks."""
    
    def __init__(self):
        """Initialize NLP pipelines."""
        self.classifier = pipeline(
            "text-classification",
            model=config.nlp["classification_model"]
        )
        self.ner = pipeline(
            "ner",
            model=config.nlp["entity_model"],
            aggregation_strategy="simple"
        )
        self.summarizer = pipeline(
            "summarization",
            model=config.nlp["summarization_model"]
        )
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=config.nlp["classification_model"]
        )

    def classify_text(self, text: str) -> ClassificationResponse:
        """
        Perform text classification.
        
        Args:
            text (str): Input text
        
        Returns:
            ClassificationResponse: Classification result
        """
        result = self.classifier(text)[0]
        return ClassificationResponse(
            label=result["label"],
            score=result["score"],
            text=text
        )

    def extract_entities(self, text: str) -> EntityResponse:
        """
        Extract entities from text.
        
        Args:
            text (str): Input text
        
        Returns:
            EntityResponse: Entity extraction result
        """
        entities = self.ner(text)
        return EntityResponse(
            entities=[{"entity": e["entity_group"], "text": e["word"]} for e in entities],
            text=text
        )

    def summarize_text(self, text: str) -> SummarizationResponse:
        """
        Summarize input text.
        
        Args:
            text (str): Input text
        
        Returns:
            SummarizationResponse: Summarization result
        """
        summary = self.summarizer(text, max_length=130, min_length=30, do_sample=False)[0]
        return SummarizationResponse(
            summary=summary["summary_text"],
            text=text
        )

    def analyze_sentiment(self, text: str) -> SentimentResponse:
        """
        Perform sentiment analysis.
        
        Args:
            text (str): Input text
        
        Returns:
            SentimentResponse: Sentiment analysis result
        """
        result = self.sentiment_analyzer(text)[0]
        return SentimentResponse(
            sentiment=result["label"],
            score=result["score"],
            text=text
        )

    def batch_process(self, texts: List[str], task: str) -> List[Dict]:
        """
        Process a batch of texts for a specific task.
        
        Args:
            texts (List[str]): List of input texts
            task (str): Task to perform (classification, ner, summarization, sentiment)
        
        Returns:
            List[Dict]: List of results
        """
        if task == "classification":
            return [self.classify_text(text).dict() for text in texts]
        elif task == "ner":
            return [self.extract_entities(text).dict() for text in texts]
        elif task == "summarization":
            return [self.summarize_text(text).dict() for text in texts]
        elif task == "sentiment":
            return [self.analyze_sentiment(text).dict() for text in texts]
        else:
            raise ValueError(f"Unsupported task: {task}")