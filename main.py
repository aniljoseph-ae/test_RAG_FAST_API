
"""
Main entry point for the NLP RAG FastAPI application.
"""
from fastapi import FastAPI
from config.config import config
from api.endpoints import router
import uvicorn
import mlflow

# Initialize FastAPI app
app = FastAPI(
    title=config.app["name"],
    description="NLP RAG API with advanced text processing capabilities"
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# MLflow setup for tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Adjust as needed
mlflow.set_experiment("nlp_rag_app")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.app["host"],
        port=config.app["port"],
        log_level=config.app["log_level"]
    )
