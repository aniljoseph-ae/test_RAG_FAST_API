
"""
FastAPI endpoints for NLP RAG application.
"""
from fastapi import FastAPI, APIRouter, HTTPException
from models.schema import (
    TextInput, BatchInput, TaskStatusResponse,
    ClassificationResponse, EntityResponse,
    SummarizationResponse, SentimentResponse
)
from services.rag_service import RAGService
from utils.cache import Cache
from utils.queue import process_nlp_task
from typing import List, Dict, Union
from uuid import uuid4

router = APIRouter()
rag_service = RAGService()
cache = Cache()

@router.post("/classify", response_model=ClassificationResponse)
async def classify_text(input: TextInput):
    """Classify text with RAG augmentation."""
    cache_key = f"classify:{input.text}"
    cached = cache.get(cache_key)
    if cached:
        return ClassificationResponse(**cached["result"])
    
    result = rag_service.process_with_rag(input.text, "classification")
    cache.set(cache_key, result)
    return ClassificationResponse(**result["result"])

@router.post("/entities", response_model=EntityResponse)
async def extract_entities(input: TextInput):
    """Extract entities with RAG augmentation."""
    cache_key = f"entities:{input.text}"
    cached = cache.get(cache_key)
    if cached:
        return EntityResponse(**cached["result"])
    
    result = rag_service.process_with_rag(input.text, "ner")
    cache.set(cache_key, result)
    return EntityResponse(**result["result"])

@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(input: TextInput):
    """Summarize text with RAG augmentation."""
    cache_key = f"summarize:{input.text}"
    cached = cache.get(cache_key)
    if cached:
        return SummarizationResponse(**cached["result"])
    
    result = rag_service.process_with_rag(input.text, "summarization")
    cache.set(cache_key, result)
    return SummarizationResponse(**result["result"])

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(input: TextInput):
    """Analyze sentiment with RAG augmentation."""
    cache_key = f"sentiment:{input.text}"
    cached = cache.get(cache_key)
    if cached:
        return SentimentResponse(**cached["result"])
    
    result = rag_service.process_with_rag(input.text, "sentiment")
    cache.set(cache_key, result)
    return SentimentResponse(**result["result"])

@router.post("/batch", response_model=List[Dict])
async def batch_process(input: BatchInput):
    """Process batch of texts synchronously."""
    results = []
    for text_input in input.texts:
        result = rag_service.process_with_rag(text_input.text, "classification")
        results.append(result)
    return results

@router.post("/async/{task}", response_model=TaskStatusResponse)
async def async_process(input: TextInput, task: str, webhook_url: str = None):
    """Process NLP task asynchronously with optional webhook."""
    task_id = str(uuid4())
    task_result = process_nlp_task.delay(input.text, task, webhook_url)
    return TaskStatusResponse(task_id=task_id, status="pending")

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of asynchronous task."""
    from celery.result import AsyncResult
    task = AsyncResult(task_id)
    if task.state == "PENDING":
        return TaskStatusResponse(task_id=task_id, status="pending")
    elif task.state == "SUCCESS":
        return TaskStatusResponse(task_id=task_id, status="completed", result=task.get())
    else:
        return TaskStatusResponse(task_id=task_id, status="failed")
