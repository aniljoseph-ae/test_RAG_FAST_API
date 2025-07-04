# NLP RAG FastAPI Application

A production-grade FastAPI application for advanced NLP tasks with Retrieval-Augmented Generation (RAG).

## Features
- **Text Classification**: Classify text into categories (e.g., positive/negative).
- **Entity Extraction**: Identify named entities in text.
- **Summarization**: Generate concise summaries of input text.
- **Sentiment Analysis**: Analyze sentiment expressed in text.
- **Batch Processing**: Process multiple texts in a single request.
- **Asynchronous Processing**: Handle long-running tasks with Celery.
- **Webhook Notifications**: Notify external systems on task completion.
- **RAG Pipeline**: Enhance NLP tasks with context from a vector database.
- **Caching**: Redis-based caching for improved performance.
- **Horizontal Scaling**: Designed for scalability with Docker and Celery.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/nlp-rag-app.git
   cd nlp-rag-app


Install dependencies:pip install -r requirements.txt


Set up environment variables in .env:WEAVIATE_API_KEY=your_weaviate_api_key
REDIS_PASSWORD=your_redis_password


Run the application:uvicorn main:app --host 0.0.0.0 --port 8000



API Endpoints

POST /api/v1/classify: Perform text classification.
Request: { "text": "Sample text" }
Response: { "label": "POSITIVE", "score": 0.95, "text": "Sample text", "timestamp": "2025-07-02T12:00:00" }


POST /api/v1/entities: Extract entities from text.
Request: { "text": "Sample text" }
Response: { "entities": [{"entity": "PERSON", "text": "John"}], "text": "Sample text", "timestamp": "2025-07-02T12:00:00" }


POST /api/v1/summarize: Summarize text.
Request: { "text": "Sample text" }
Response: { "summary": "Summary text", "text": "Sample text", "timestamp": "2025-07-02T12:00:00" }


POST /api/v1/sentiment: Analyze sentiment.
Request: { "text": "Sample text" }
Response: { "sentiment": "POSITIVE", "score": 0.95, "text": "Sample text", "timestamp": "2025-07-02T12:00:00" }


POST /api/v1/batch: Process batch of texts.
Request: { "texts": [{"text": "Text 1"}, {"text": "Text 2"}] }
Response: [ { "result": {...}, "context": [...] }, ... ]


POST /api/v1/async/{task}: Submit async task (classification, ner, summarization, sentiment).
Request: { "text": "Sample text", "webhook_url": "http://example.com/webhook" }
Response: { "task_id": "uuid", "status": "pending" }


GET /api/v1/status/{task_id}: Check async task status.
Response: { "task_id": "uuid", "status": "completed", "result": {...} }



RAG Implementation
The RAG pipeline enhances NLP tasks by:

Embedding: Uses sentence-transformers/all-MiniLM-L6-v2 to create domain-specific embeddings.
Storage: Stores documents in Weaviate vector database.
Retrieval: Retrieves similar documents based on cosine similarity of embeddings.
Reranking: Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank retrieved documents.
Context Augmentation: Combines top relevant documents with input text for NLP processing.
Storage for Future Retrieval: Saves processed documents to improve future retrieval.

Deployment

Docker: Use the provided Dockerfile to build and deploy.
CI/CD: GitHub Actions pipeline (ci.yml) for automated testing and deployment.
MLflow: Tracks experiments and model performance.

Scaling

Horizontal Scaling: Deploy multiple Docker containers with load balancing.
Task Queue: Celery with Redis backend for async task processing.
Caching: Redis-based caching to reduce redundant computations.


