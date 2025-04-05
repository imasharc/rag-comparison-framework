"""
Vector store module for storing and retrieving document embeddings.
"""
from typing import List, Dict, Any, Optional, Type, Union
import logging
import os
import pickle
from abc import ABC, abstractmethod

from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VectorStoreManager(ABC):
    """Abstract base class for vector store managers."""
    
    @abstractmethod
    def create_vector_store(self, documents: List[Document]) -> Any:
        """
        Create a vector store from documents.
        
        Args:
            documents (List[Document]): Documents to create the vector store from
            
        Returns:
            Any: The created vector store
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path (str): Path to save the vector store to
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load the vector store from disk.
        
        Args:
            path (str): Path to load the vector store from
        """
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query (str): Query string
            k (int): Number of documents to return
            
        Returns:
            List[Document]: List of similar documents
        """
        pass


class FAISSVectorStoreManager(VectorStoreManager):
    """Vector store manager implementation using FAISS."""
    
    def __init__(self, embedding_model: Optional[Embeddings] = None):
        """
        Initialize the FAISS vector store manager.
        
        Args:
            embedding_model (Optional[Embeddings]): Embedding model to use
        """
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store = None
    
    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from documents.
        
        Args:
            documents (List[Document]): Documents to create the vector store from
            
        Returns:
            FAISS: The created FAISS vector store
            
        Raises:
            ValueError: If the vector store creation fails
        """
        try:
            logger.info(f"Creating FAISS vector store from {len(documents)} documents")
            self.vector_store = FAISS.from_documents(documents, self.embedding_model)
            logger.info("FAISS vector store created successfully")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error creating FAISS vector store: {str(e)}")
            raise ValueError(f"Failed to create FAISS vector store: {str(e)}")
    
    def save(self, path: str) -> None:
        """
        Save the FAISS vector store to disk.
        
        Args:
            path (str): Path to save the vector store to
            
        Raises:
            ValueError: If the vector store hasn't been created yet or if saving fails
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store has not been created yet")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            logger.info(f"Saving FAISS vector store to {path}")
            self.vector_store.save_local(path)
            logger.info("FAISS vector store saved successfully")
        except Exception as e:
            logger.error(f"Error saving FAISS vector store: {str(e)}")
            raise ValueError(f"Failed to save FAISS vector store: {str(e)}")
    
    def load(self, path: str) -> None:
        """
        Load the FAISS vector store from disk.
        
        Args:
            path (str): Path to load the vector store from
            
        Raises:
            FileNotFoundError: If the vector store file doesn't exist
            ValueError: If loading fails
        """
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Vector store file not found: {path}")
            
            logger.info(f"Loading FAISS vector store from {path}")
            # Added allow_dangerous_deserialization parameter for security confirmation
            self.vector_store = FAISS.load_local(
                path, 
                self.embedding_model,
                allow_dangerous_deserialization=True  # Safe because we created this file ourselves
            )
            logger.info("FAISS vector store loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error loading FAISS vector store: {str(e)}")
            raise ValueError(f"Failed to load FAISS vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search on the FAISS vector store.
        
        Args:
            query (str): Query string
            k (int): Number of documents to return
            
        Returns:
            List[Document]: List of similar documents
            
        Raises:
            ValueError: If the vector store hasn't been created yet or if search fails
        """
        try:
            if self.vector_store is None:
                raise ValueError("Vector store has not been created yet")
            
            logger.info(f"Performing similarity search for query: {query}")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise ValueError(f"Failed to perform similarity search: {str(e)}")


class VectorStoreFactory:
    """Factory for creating vector store managers."""
    
    @staticmethod
    def get_vector_store_manager(store_type: str = "faiss", **kwargs) -> VectorStoreManager:
        """
        Get a vector store manager based on type and parameters.
        
        Args:
            store_type (str): Type of vector store manager to create
            **kwargs: Additional parameters for the vector store manager
            
        Returns:
            VectorStoreManager: An instance of a vector store manager
            
        Raises:
            ValueError: If the vector store type is not supported
        """
        if store_type == "faiss":
            return FAISSVectorStoreManager(**kwargs)
        else:
            logger.error(f"Unsupported vector store type: {store_type}")
            raise ValueError(f"Unsupported vector store type: {store_type}")