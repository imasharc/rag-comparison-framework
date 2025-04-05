"""
Test script to verify OpenAI API key from .env file.
This script loads the API key from your .env file and tests if it can:
1. Connect to the OpenAI API
2. List available models
3. Generate embeddings with text-embedding-ada-002
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_openai_key():
    """Test if the OpenAI API key from .env file is valid and has proper permissions."""
    
    # Step 1: Load environment variables from .env file
    logger.info("Loading environment variables from .env file...")
    load_dotenv()
    
    # Step 2: Get the API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        logger.error("No OPENAI_API_KEY found in .env file")
        return False
    
    # Log the first few characters of the API key (for debugging prefix)
    key_prefix = api_key[:7] + "..." if len(api_key) > 10 else "too short"
    logger.info(f"API key found with prefix: {key_prefix}")
    
    # Step 3: Create OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Step 4: Test basic API connection by listing models
    try:
        logger.info("Testing API connection by listing models...")
        models = client.models.list()
        logger.info(f"Successfully connected to OpenAI API. Found {len(models.data)} models.")
        
        # Optional: Print available models
        logger.info("Available models:")
        for model in models.data[:5]:  # Show just first 5 to avoid cluttering output
            logger.info(f"  - {model.id}")
        
        if len(models.data) > 5:
            logger.info(f"  - ...and {len(models.data) - 5} more")
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {str(e)}")
        return False
    
    # Step 5: Test embeddings specifically
    try:
        logger.info("Testing embedding model access...")
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input="This is a test of the embedding model."
        )
        
        embedding = response.data[0].embedding
        embedding_length = len(embedding)
        logger.info(f"Successfully generated embeddings of dimension {embedding_length}")
        return True
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting OpenAI API key test...")
    success = test_openai_key()
    
    if success:
        logger.info("✅ All tests passed! Your OpenAI API key is working correctly.")
        sys.exit(0)
    else:
        logger.error("❌ Test failed! Please check your API key and permissions.")
        sys.exit(1)