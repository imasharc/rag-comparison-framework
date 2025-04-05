"""
Document loader module that handles loading different types of documents.
"""
from typing import List, Optional, Protocol, Dict, Any, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.docstore.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a file path.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            List[Document]: List of loaded documents
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file type is not supported
        """
        pass


class PDFDocumentLoader(DocumentLoader):
    """Loader for PDF documents."""
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a PDF file.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            List[Document]: List of loaded documents
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If there's an issue parsing the PDF
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            logger.info(f"Loading PDF document: {file_path}")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF document: {str(e)}")
            raise ValueError(f"Failed to load PDF document: {str(e)}")


class TextDocumentLoader(DocumentLoader):
    """Loader for plain text documents."""
    
    def load(self, file_path: str) -> List[Document]:
        """
        Load documents from a text file.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            List[Document]: List of loaded documents
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If there's an issue parsing the text file
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            logger.info(f"Loading text document: {file_path}")
            loader = TextLoader(file_path)
            documents = loader.load()
            logger.info(f"Loaded text document with {len(documents)} document(s)")
            return documents
        except Exception as e:
            logger.error(f"Error loading text document: {str(e)}")
            raise ValueError(f"Failed to load text document: {str(e)}")


class DocumentLoaderFactory:
    """Factory for creating document loaders based on file type."""
    
    @staticmethod
    def get_loader(file_path: str) -> DocumentLoader:
        """
        Get the appropriate document loader based on file extension.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            DocumentLoader: An instance of a document loader
            
        Raises:
            ValueError: If the file type is not supported
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return PDFDocumentLoader()
        elif file_extension == '.txt':
            return TextDocumentLoader()
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            raise ValueError(f"Unsupported file type: {file_extension}")


def load_document(file_path: str) -> List[Document]:
    """
    Load a document using the appropriate loader.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        List[Document]: List of loaded documents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file type is not supported or there's an issue loading
    """
    try:
        loader = DocumentLoaderFactory.get_loader(file_path)
        return loader.load(file_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error in load_document: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in load_document: {str(e)}")
        raise ValueError(f"Failed to load document: {str(e)}")