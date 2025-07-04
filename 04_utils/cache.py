
"""
Redis-based caching system for API responses.
"""
import redis
import json
from typing import Optional, Any
from config.config import config

class Cache:
    """Redis cache for storing API responses."""
    
    def __init__(self):
        """Initialize Redis client."""
        self.client = redis.Redis(
            host=config.cache["host"],
            port=config.cache["port"],
            db=config.cache["db"],
            password=os.getenv("REDIS_PASSWORD")
        )
        self.ttl = config.cache["ttl"]

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve cached value by key.
        
        Args:
            key (str): Cache key
        
        Returns:
            Optional[Any]: Cached value or None
        """
        value = self.client.get(key)
        return json.loads(value) if value else None

    def set(self, key: str, value: Any):
        """
        Store value in cache with TTL.
        
        Args:
            key (str): Cache key
            value: Value to cache
        """
        self.client.setex(key, self.ttl, json.dumps(value))
