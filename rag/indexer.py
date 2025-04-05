"""
Indexer module for chunking documents and preparing them for embedding.
"""
from typing import List, Dict, Any, Optional
import logging
from abc import ABC, abstractmethod

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentIndexer(ABC):
    """Abstract base class for document indexers."""
    
    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of document chunks
        """
        pass


class ChunkingDocumentIndexer(DocumentIndexer):
    """Indexer that splits documents into chunks of specified size."""
    
    def __init__(self, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200,
                separators: Optional[List[str]] = None):
        """
        Initialize the chunking document indexer.
        
        Args:
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between chunks in characters
            separators (Optional[List[str]]): List of separators to use for chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks using RecursiveCharacterTextSplitter.
        
        Args:
            documents (List[Document]): List of documents to split
            
        Returns:
            List[Document]: List of document chunks
            
        Raises:
            ValueError: If the chunking process fails
        """
        try:
            logger.info(f"Splitting {len(documents)} documents into chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from documents")
            return chunks
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise ValueError(f"Failed to split documents: {str(e)}")


class IndexerFactory:
    """Factory for creating document indexers."""
    
    @staticmethod
    def get_indexer(indexer_type: str = "chunking", **kwargs) -> DocumentIndexer:
        """
        Get an indexer based on type and parameters.
        
        Args:
            indexer_type (str): Type of indexer to create
            **kwargs: Additional parameters for the indexer
            
        Returns:
            DocumentIndexer: An instance of a document indexer
            
        Raises:
            ValueError: If the indexer type is not supported
        """
        if indexer_type == "chunking":
            return ChunkingDocumentIndexer(**kwargs)
        else:
            logger.error(f"Unsupported indexer type: {indexer_type}")
            raise ValueError(f"Unsupported indexer type: {indexer_type}")


def index_documents(documents: List[Document], 
                   indexer_type: str = "chunking", 
                   **kwargs) -> List[Document]:
    """
    Index documents using the specified indexer.
    
    Args:
        documents (List[Document]): List of documents to index
        indexer_type (str): Type of indexer to use
        **kwargs: Additional parameters for the indexer
        
    Returns:
        List[Document]: List of indexed document chunks
        
    Raises:
        ValueError: If the indexing process fails
    """
    try:
        indexer = IndexerFactory.get_indexer(indexer_type, **kwargs)
        return indexer.split_documents(documents)
    except ValueError as e:
        logger.error(f"Error in index_documents: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in index_documents: {str(e)}")
        raise ValueError(f"Failed to index documents: {str(e)}")