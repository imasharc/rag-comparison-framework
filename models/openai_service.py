"""
OpenAI service module for interacting with the OpenAI API.
"""
from typing import List, Dict, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from openai.types.chat import ChatCompletion

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LLMService(ABC):
    """Abstract base class for LLM services."""
    
    @abstractmethod
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str,
                     **kwargs) -> str:
        """
        Generate text using the LLM.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            **kwargs: Additional parameters for the LLM
            
        Returns:
            str: Generated text
        """
        pass


class OpenAIService(LLMService):
    """LLM service implementation using OpenAI."""
    
    def __init__(self, 
                api_key: str,
                model: str = "gpt-4o-mini",
                max_tokens: int = 1000,
                temperature: float = 0.7):
        """
        Initialize the OpenAI service.
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model to use
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature for generation
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
    
    def generate_text(self, 
                     system_prompt: str, 
                     user_prompt: str,
                     temperature: Optional[float] = None,
                     max_tokens: Optional[int] = None,
                     **kwargs) -> str:
        """
        Generate text using the OpenAI API.
        
        Args:
            system_prompt (str): System prompt
            user_prompt (str): User prompt
            temperature (Optional[float]): Temperature for generation
            max_tokens (Optional[int]): Maximum number of tokens to generate
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: Generated text
            
        Raises:
            ValueError: If the API call fails
        """
        try:
            # Override default parameters with any provided explicitly
            model = kwargs.get('model', self.model)
            max_tokens_to_use = max_tokens if max_tokens is not None else self.max_tokens
            temperature_to_use = temperature if temperature is not None else self.temperature
            
            logger.info(f"Generating text with model {model}")
            
            response: ChatCompletion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens_to_use,
                temperature=temperature_to_use
            )
            
            generated_text = response.choices[0].message.content
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {str(e)}")
            raise ValueError(f"Failed to generate text with OpenAI: {str(e)}")
    
    def update_api_key(self, api_key: str) -> None:
        """
        Update the API key.
        
        Args:
            api_key (str): New API key to use
            
        Raises:
            ValueError: If the new API key is invalid
        """
        try:
            # Create a new client with the new API key
            new_client = OpenAI(api_key=api_key)
            
            # Test the key with a simple API call
            new_client.models.list()
            
            # If successful, update the client
            self.client = new_client
            logger.info("API key updated successfully")
        except Exception as e:
            logger.error(f"Error updating API key: {str(e)}")
            raise ValueError(f"Invalid API key: {str(e)}")