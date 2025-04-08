"""
API routes for the Flask application.
"""
from typing import Dict, Any, Optional
import logging
from flask import Blueprint, request, jsonify, current_app

from utils.error_handler import api_error_handler, BadRequestError, NotFoundError
from models.openai_service import OpenAIService
from rag.query_engine import QueryEngine

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('api', __name__)


@api_bp.route('/', methods=['GET'])
@api_error_handler
def index():
    """
    Root endpoint.
    
    Returns:
        Dict[str, str]: Status message
    """
    return jsonify({"status": "API is running"})


@api_bp.route('/query', methods=['POST'])
@api_error_handler
def process_query():
    """
    Process a query using the query engine.
    
    Returns:
        Dict[str, str]: Response with generated answer
        
    Raises:
        BadRequestError: If no query is provided
    """
    # Extract data from request
    data = request.json
    query = data.get('query', '')
    
    if not query:
        raise BadRequestError("No query provided")
    
    # Get the query engine from the application context
    query_engine = current_app.config.get('QUERY_ENGINE')
    
    if not query_engine:
        logger.error("Query engine not found in application context")
        raise NotFoundError("Query engine not available")
    
    # Process the query
    try:
        response = query_engine.query(query)
        return jsonify({'response': response})
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise BadRequestError(f"Error processing query: {str(e)}")


@api_bp.route('/config', methods=['POST'])
@api_error_handler
def configure_api():
    """
    Configure the OpenAI API key.
    
    Returns:
        Dict[str, str]: Status message
        
    Raises:
        BadRequestError: If no API key is provided or if the key is invalid
    """
    data = request.json
    
    # Check if we should restore the default key
    if data.get('use_default', False):
        openai_service = current_app.config.get('OPENAI_SERVICE')
        default_api_key = current_app.config.get('DEFAULT_API_KEY')
        
        if not openai_service or not default_api_key:
            raise NotFoundError("OpenAI service or default API key not available")
        
        try:
            openai_service.update_api_key(default_api_key)
            return jsonify({'status': 'success', 'message': 'Restored default API key'})
        except Exception as e:
            logger.error(f"Error restoring default API key: {str(e)}")
            raise BadRequestError(f"Error restoring default API key: {str(e)}")
    
    # Get the new API key
    new_api_key = data.get('api_key')
    
    if not new_api_key:
        raise BadRequestError("No API key provided")
    
    # Get the OpenAI service from the application context
    openai_service = current_app.config.get('OPENAI_SERVICE')
    
    if not openai_service:
        raise NotFoundError("OpenAI service not available")
    
    # Update the API key
    try:
        openai_service.update_api_key(new_api_key)
        return jsonify({'status': 'success', 'message': 'API key configured successfully'})
    except Exception as e:
        logger.error(f"Error configuring API key: {str(e)}")
        raise BadRequestError(f"Invalid API key: {str(e)}")

@api_bp.route('/complete', methods=['POST'])
@api_error_handler
def complete_text():
    """
    Generate text using the OpenAI API.
    
    Returns:
        Dict[str, str]: Response with generated text
        
    Raises:
        BadRequestError: If required parameters are missing
    """
    # Extract data from request
    data = request.json
    system_prompt = data.get('system_prompt', '')
    user_prompt = data.get('user_prompt', '')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens', 500)
    
    if not system_prompt or not user_prompt:
        raise BadRequestError("System prompt and user prompt are required")
    
    # Get the OpenAI service from the application context
    openai_service = current_app.config.get('OPENAI_SERVICE')
    
    if not openai_service:
        logger.error("OpenAI service not found in application context")
        raise NotFoundError("OpenAI service not available")
    
    # Generate text
    try:
        text = openai_service.generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return jsonify({'text': text})
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise BadRequestError(f"Error generating text: {str(e)}")