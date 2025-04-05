"""
RAG (Retrieval-Augmented Generation) module.
"""
from typing import List, Dict, Any, Optional, Union
import logging
import os

from langchain.docstore.document import Document

from rag.document_loader import load_document
from rag.indexer import index_documents
from rag.vector_store import VectorStoreFactory
from rag.retriever import get_retriever
from rag.query_engine import get_query_engine
from models.openai_service import OpenAIService

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def initialize_rag_system(
    document_path: str,
    openai_service: OpenAIService,
    vector_store_path: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    system_prompt_template: Optional[str] = None
) -> Any:
    """
    Initialize the RAG system.
    
    Args:
        document_path (str): Path to the document to index
        openai_service (OpenAIService): OpenAI service for embeddings and generation
        vector_store_path (Optional[str]): Path to save/load the vector store
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters
        system_prompt_template (Optional[str]): Template for the system prompt
        
    Returns:
        Any: The query engine
        
    Raises:
        ValueError: If initialization fails
    """
    try:
        # Create vector store manager
        vector_store_manager = VectorStoreFactory.get_vector_store_manager(
            store_type="faiss",
            embedding_model=None  # Use default OpenAI embeddings
        )
        
        # If vector store exists and path is provided, load it
        if vector_store_path and os.path.exists(vector_store_path):
            logger.info(f"Loading existing vector store from {vector_store_path}")
            vector_store_manager.load(vector_store_path)
        else:
            # Otherwise, create a new vector store
            logger.info(f"Creating new vector store from {document_path}")
            
            # Load document
            documents = load_document(document_path)
            
            # Chunk documents
            document_chunks = index_documents(
                documents,
                indexer_type="chunking",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Create vector store
            vector_store_manager.create_vector_store(document_chunks)
            
            # Save vector store if path is provided
            if vector_store_path:
                logger.info(f"Saving vector store to {vector_store_path}")
                os.makedirs(os.path.dirname(vector_store_path), exist_ok=True)
                vector_store_manager.save(vector_store_path)
        
        # Create retriever
        retriever = get_retriever(
            retriever_type="vector_store",
            vector_store_manager=vector_store_manager
        )
        
        # Create query engine
        query_engine = get_query_engine(
            engine_type="rag",
            retriever=retriever,
            llm_service=openai_service,
            system_prompt_template=system_prompt_template
        )
        
        logger.info("RAG system initialized successfully")
        return query_engine
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        raise ValueError(f"Failed to initialize RAG system: {str(e)}")