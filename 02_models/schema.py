
"""
Pydantic models for API request and response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class TextInput(BaseModel):
    """Model for text input to NLP tasks."""
    text: str = Field(..., min_length=1, description="Text to process")
    metadata: Optional[Dict] = Field(default=None, description="Optional metadata")

class BatchInput(BaseModel):
    """Model for batch text input."""
    texts: List[TextInput] = Field(..., min_items=1, description="List of texts to process")

class ClassificationResponse(BaseModel):
    """Response model for text classification."""
    label: str
    score: float
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EntityResponse(BaseModel):
    """Response model for entity extraction."""
    entities: List[Dict[str, str]]
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SummarizationResponse(BaseModel):
    """Response model for text summarization."""
    summary: str
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str
    score: float
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class TaskStatusResponse(BaseModel):
    """Response model for async task status."""
    task_id: str
    status: str
    result: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class WebhookPayload(BaseModel):
    """Model for webhook notification payload."""
    task_id: str
    status: str
    result: Dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)
