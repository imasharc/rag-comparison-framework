"""
Retriever module for retrieving relevant documents from the vector store.
"""
from typing import List, Dict, Any, Optional, Union
import logging
from abc import ABC, abstractmethod

from langchain.docstore.document import Document

from rag.vector_store import VectorStoreManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentRetriever(ABC):
    """Abstract base class for document retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query (str): Query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
        """
        pass


class VectorStoreRetriever(DocumentRetriever):
    """Retriever implementation using a vector store."""
    
    def __init__(self, vector_store_manager: VectorStoreManager):
        """
        Initialize the vector store retriever.
        
        Args:
            vector_store_manager (VectorStoreManager): Vector store manager to use
        """
        self.vector_store_manager = vector_store_manager
    
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query using the vector store.
        
        Args:
            query (str): Query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Document]: List of retrieved documents
            
        Raises:
            ValueError: If retrieval fails
        """
        try:
            logger.info(f"Retrieving documents for query: {query}")
            documents = self.vector_store_manager.similarity_search(query, k=k)
            logger.info(f"Retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise ValueError(f"Failed to retrieve documents: {str(e)}")


class RetrieverFactory:
    """Factory for creating document retrievers."""
    
    @staticmethod
    def get_retriever(retriever_type: str = "vector_store", 
                     vector_store_manager: Optional[VectorStoreManager] = None, 
                     **kwargs) -> DocumentRetriever:
        """
        Get a document retriever based on type and parameters.
        
        Args:
            retriever_type (str): Type of retriever to create
            vector_store_manager (Optional[VectorStoreManager]): Vector store manager to use
            **kwargs: Additional parameters for the retriever
            
        Returns:
            DocumentRetriever: An instance of a document retriever
            
        Raises:
            ValueError: If the retriever type is not supported or required parameters are missing
        """
        if retriever_type == "vector_store":
            if vector_store_manager is None:
                raise ValueError("Vector store manager is required for vector store retriever")
            return VectorStoreRetriever(vector_store_manager)
        else:
            logger.error(f"Unsupported retriever type: {retriever_type}")
            raise ValueError(f"Unsupported retriever type: {retriever_type}")


def get_retriever(retriever_type: str = "vector_store", 
                 vector_store_manager: Optional[VectorStoreManager] = None, 
                 **kwargs) -> DocumentRetriever:
    """
    Get a document retriever.
    
    Args:
        retriever_type (str): Type of retriever to create
        vector_store_manager (Optional[VectorStoreManager]): Vector store manager to use
        **kwargs: Additional parameters for the retriever
        
    Returns:
        DocumentRetriever: An instance of a document retriever
        
    Raises:
        ValueError: If the retriever creation fails
    """
    try:
        return RetrieverFactory.get_retriever(
            retriever_type=retriever_type,
            vector_store_manager=vector_store_manager,
            **kwargs
        )
    except ValueError as e:
        logger.error(f"Error in get_retriever: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_retriever: {str(e)}")
        raise ValueError(f"Failed to get retriever: {str(e)}")