"""
Error handling utilities.
"""
from typing import Dict, Any, Optional, Callable, Type, Union, Tuple
import logging
import functools
import traceback
from flask import jsonify, Response

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Base class for API errors."""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the API error.
        
        Args:
            message (str): Error message
            status_code (int): HTTP status code
            details (Optional[Dict[str, Any]]): Additional error details
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class BadRequestError(APIError):
    """Error for bad requests."""
    
    def __init__(self, message: str = "Bad request", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the bad request error.
        
        Args:
            message (str): Error message
            details (Optional[Dict[str, Any]]): Additional error details
        """
        super().__init__(message, 400, details)


class NotFoundError(APIError):
    """Error for resources not found."""
    
    def __init__(self, message: str = "Resource not found", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the not found error.
        
        Args:
            message (str): Error message
            details (Optional[Dict[str, Any]]): Additional error details
        """
        super().__init__(message, 404, details)


class InternalServerError(APIError):
    """Error for internal server errors."""
    
    def __init__(self, message: str = "Internal server error", details: Optional[Dict[str, Any]] = None):
        """
        Initialize the internal server error.
        
        Args:
            message (str): Error message
            details (Optional[Dict[str, Any]]): Additional error details
        """
        super().__init__(message, 500, details)


def handle_api_error(error: APIError) -> Tuple[Response, int]:
    """
    Handle an API error.
    
    Args:
        error (APIError): The API error to handle
        
    Returns:
        Tuple[Response, int]: Flask response and status code
    """
    logger.error(f"API Error: {error.message}")
    if error.details:
        logger.error(f"Error details: {error.details}")
    
    response = jsonify({
        "error": error.message,
        "details": error.details
    })
    
    return response, error.status_code


def api_error_handler(func: Callable) -> Callable:
    """
    Decorator to handle API errors.
    
    Args:
        func (Callable): Function to decorate
        
    Returns:
        Callable: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIError as e:
            return handle_api_error(e)
        except Exception as e:
            logger.error(f"Unhandled exception: {str(e)}")
            logger.error(traceback.format_exc())
            return handle_api_error(InternalServerError(str(e)))
    
    return wrapper