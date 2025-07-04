
"""
Configuration loader for the NLP RAG App.
Loads settings from config.yaml and environment variables.
"""
from pathlib import Path
import yaml
from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class to hold application settings."""
    
    def __init__(self):
        """Initialize configuration by loading from YAML file."""
        config_path = Path(__file__).parent / "config.yaml"
        
        try:
            with config_path.open("r") as file:
                self.config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing config.yaml: {e}")

    def get(self, section: str) -> Dict[str, Any]:
        """
        Retrieve configuration section.
        
        Args:
            section (str): Configuration section to retrieve (e.g., 'app', 'vector_db')
        
        Returns:
            Dict[str, Any]: Configuration section as dictionary
        """
        return self.config.get(section, {})

    @property
    def app(self) -> Dict[str, Any]:
        """Get application configuration."""
        return self.get("app")

    @property
    def vector_db(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return self.get("vector_db")

    @property
    def nlp(self) -> Dict[str, Any]:
        """Get NLP model configuration."""
        return self.get("nlp")

    @property
    def cache(self) -> Dict[str, Any]:
        """Get cache configuration."""
        return self.get("cache")

    @property
    def queue(self) -> Dict[str, Any]:
        """Get task queue configuration."""
        return self.get("queue")

    @property
    def webhook(self) -> Dict[str, Any]:
        """Get webhook configuration."""
        return self.get("webhook")

# Singleton instance of Config
config = Config()
