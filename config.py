"""
Configuration settings for the application.
"""
import os
from typing import Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Configure simple constants
USE_LOCAL_EMBEDDINGS = True  # Set to True to use local embeddings instead of OpenAI

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# API configuration
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "5000"))

# RAG configuration
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", os.path.join(BASE_DIR, "data", "NovaTech_Security_Policy.pdf"))
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", os.path.join(BASE_DIR, "data", "vector_store"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# System prompt template for RAG
RAG_SYSTEM_PROMPT_TEMPLATE = os.getenv(
    "RAG_SYSTEM_PROMPT_TEMPLATE",
    "You are a helpful assistant that answers questions about NovaTech Dynamics' security policy. "
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "Use the information from the context exclusively and avoid adding information not present in the context. "
    "Keep the answer concise but comprehensive.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

# Create necessary directories
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)

# Function to get all configuration as a dictionary
def get_config() -> Dict[str, Any]:
    """
    Get all configuration settings as a dictionary.
    
    Returns:
        Dict[str, Any]: Configuration settings
    """
    return {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "OPENAI_MODEL": OPENAI_MODEL,
        "API_HOST": API_HOST,
        "API_PORT": API_PORT,
        "DOCUMENT_PATH": DOCUMENT_PATH,
        "VECTOR_STORE_PATH": VECTOR_STORE_PATH,
        "CHUNK_SIZE": CHUNK_SIZE,
        "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "RAG_SYSTEM_PROMPT_TEMPLATE": RAG_SYSTEM_PROMPT_TEMPLATE
    }