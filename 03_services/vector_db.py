
"""
Vector database integration for RAG using Weaviate.
Handles document storage, embedding, and retrieval.
"""
import weaviate
import os
from typing import List, Dict, Any
from config.config import config
from sentence_transformers import SentenceTransformer

class VectorDB:
    """Vector database client for RAG operations."""
    
    def __init__(self):
        """Initialize Weaviate client and embedding model."""
        self.client = weaviate.Client(
            url=config.vector_db["url"],
            auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))
        )
        self.embedding_model = SentenceTransformer(config.vector_db["embedding_model"])
        self.class_name = config.vector_db["class_name"]

        # Ensure schema exists
        self._create_schema()

    def _create_schema(self):
        """Create Weaviate schema if it doesn't exist."""
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # We'll provide our own embeddings
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["object"]}
            ]
        }
        if not self.client.schema.contains(schema):
            self.client.schema.create_class(schema)

    def store_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """
        Store a document in the vector database.
        
        Args:
            text (str): Text to store
            metadata (Dict[str, Any]): Optional metadata
        
        Returns:
            str: Document ID
        """
        vector = self.embedding_model.encode(text).tolist()
        data_object = {
            "text": text,
            "metadata": metadata or {}
        }
        return self.client.data_object.create(
            data_object=data_object,
            class_name=self.class_name,
            vector=vector
        )

    def retrieve_similar(self, text: str, limit: int = 5) -> List[Dict]:
        """
        Retrieve similar documents based on text input.
        
        Args:
            text (str): Query text
            limit (int): Number of results to return
        
        Returns:
            List[Dict]: List of similar documents with scores
        """
        vector = self.embedding_model.encode(text).tolist()
        results = self.client.query.get(
            self.class_name, ["text", "metadata"]
        ).with_near_vector({
            "vector": vector
        }).with_limit(limit).do()
        
        return results["data"]["Get"][self.class_name]
