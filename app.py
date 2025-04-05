"""
Main Flask application module.
"""
from typing import Dict, Any, Optional
import logging
import os
from flask import Flask
from flask_cors import CORS

from config import get_config, OPENAI_API_KEY, API_HOST, API_PORT, DOCUMENT_PATH, VECTOR_STORE_PATH
from config import CHUNK_SIZE, CHUNK_OVERLAP, RAG_SYSTEM_PROMPT_TEMPLATE, OPENAI_MODEL
from models.openai_service import OpenAIService
from rag import initialize_rag_system
from api.routes import api_bp

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_app(config_override: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config_override (Optional[Dict[str, Any]]): Override configuration settings
        
    Returns:
        Flask: The configured Flask application
    """
    app = Flask(__name__)
    
    # Enable CORS for all routes
    CORS(app)
    
    # Load configuration
    config = get_config()
    if config_override:
        config.update(config_override)
    
    # Add configuration to app
    for key, value in config.items():
        app.config[key] = value
    
    # Initialize OpenAI service
    openai_service = OpenAIService(
        api_key=config['OPENAI_API_KEY'],
        model=config['OPENAI_MODEL']
    )
    
    # Store OpenAI service in app context
    app.config['OPENAI_SERVICE'] = openai_service
    app.config['DEFAULT_API_KEY'] = config['OPENAI_API_KEY']
    
    # Initialize RAG system
    try:
        # Check if document exists
        if not os.path.exists(config['DOCUMENT_PATH']):
            logger.warning(f"Document not found: {config['DOCUMENT_PATH']}")
            logger.warning("RAG system will not be initialized")
        else:
            query_engine = initialize_rag_system(
                document_path=config['DOCUMENT_PATH'],
                openai_service=openai_service,
                vector_store_path=config['VECTOR_STORE_PATH'],
                chunk_size=config['CHUNK_SIZE'],
                chunk_overlap=config['CHUNK_OVERLAP'],
                system_prompt_template=config['RAG_SYSTEM_PROMPT_TEMPLATE']
            )
            
            # Store query engine in app context
            app.config['QUERY_ENGINE'] = query_engine
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        logger.warning("Continuing without RAG capabilities")
    
    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    
    return app


if __name__ == '__main__':
    app = create_app()
    
    # Log configuration
    logger.info(f"Starting API server at http://{API_HOST}:{API_PORT}")
    logger.info(f"Using model: {OPENAI_MODEL}")
    
    # Run the application
    app.run(host=API_HOST, port=API_PORT, debug=True)