"""
Base class for all RAG variants.
"""
from abc import ABC, abstractmethod
import logging

from rag_client import RAGClient

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGVariant(ABC):
    """Base class for all RAG variants."""
    
    def __init__(self, name: str, rag_client: RAGClient):
        """
        Initialize the RAG variant.
        
        Args:
            name (str): Name of the variant
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        self.name = name
        self.client = rag_client
        logger.info(f"Initialized RAG variant: {name}")
    
    @abstractmethod
    def query(self, question: str) -> str:
        """
        Process a query and return a response.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The response
        """
        pass
    
    def get_name(self) -> str:
        """
        Get the name of this RAG variant.
        
        Returns:
            str: The name of the variant
        """
        return self.name


class BaselineRAG(RAGVariant):
    """The baseline RAG implementation."""
    
    def __init__(self, rag_client: RAGClient):
        """
        Initialize the baseline RAG variant.
        
        Args:
            rag_client (RAGClient): Client for interacting with the baseline RAG API
        """
        super().__init__("Baseline RAG", rag_client)
    
    def query(self, question: str) -> str:
        """
        Process a query using the baseline RAG system.
        
        Args:
            question (str): The question to answer
            
        Returns:
            str: The response from the baseline RAG system
        """
        logger.info(f"Processing query with baseline RAG: {question}")
        return self.client.query(question)