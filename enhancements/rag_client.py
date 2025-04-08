"""
Client for interacting with the baseline RAG API.
"""
import requests
import json
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGClient:
    """Client for interacting with the baseline RAG API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:5000/api"):
        """
        Initialize the RAG client.
        
        Args:
            base_url (str): Base URL of the RAG API
        """
        self.base_url = base_url
        logger.info(f"Initialized RAG client for {base_url}")
    
    def query(self, question: str) -> str:
        """
        Send a query to the baseline RAG system.
        
        Args:
            question (str): The question to ask
            
        Returns:
            str: The response from the RAG system
        """
        try:
            url = f"{self.base_url}/query"
            payload = {"query": question}
            
            logger.info(f"Sending query to baseline RAG: {question}")
            response = requests.post(
                url=url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            
            response.raise_for_status()  # Raise exception for non-200 responses
            result = response.json()
            
            return result.get("response", "No response received")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying baseline RAG: {str(e)}")
            return f"Error: {str(e)}"
    
    def get_openai_completion(self, 
                              system_prompt: str, 
                              user_prompt: str, 
                              temperature: float = 0.7,
                              max_tokens: int = 500) -> str:
        """
        Get a completion from the OpenAI API via the baseline system.
        
        This allows the enhanced RAG variants to use the same OpenAI integration
        as the baseline system without duplicating API keys or configuration.
        
        Args:
            system_prompt (str): The system prompt
            user_prompt (str): The user prompt
            temperature (float): The temperature for generation
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated text
        """
        try:
            # Call the complete endpoint of the baseline API
            url = f"{self.base_url}/complete"
            payload = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = requests.post(
                url=url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("text", "No completion received")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting OpenAI completion: {str(e)}")
            return f"Error: {str(e)}"