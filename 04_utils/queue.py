
"""
Celery-based task queue for asynchronous NLP processing.
"""
from celery import Celery
from config.config import config
from services.rag_service import RAGService
import requests

app = Celery(
    "nlp_tasks",
    broker=config.queue["broker_url"],
    backend=config.queue["backend_url"]
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True
)

@app.task
def process_nlp_task(text: str, task: str, webhook_url: str = None) -> dict:
    """
    Process NLP task asynchronously.
    
    Args:
        text (str): Input text
        task (str): NLP task to perform
        webhook_url (str, optional): Webhook URL for notification
    
    Returns:
        dict: Task result
    """
    rag_service = RAGService()
    result = rag_service.process_with_rag(text, task)
    
    # Send webhook notification if configured
    if webhook_url and config.webhook["enabled"]:
        payload = {
            "task_id": process_nlp_task.request.id,
            "status": "completed",
            "result": result,
            "timestamp": str(datetime.utcnow())
        }
        try:
            requests.post(webhook_url, json=payload, timeout=config.webhook["timeout"])
        except requests.RequestException:
            pass  # Log error in production
    
    return result