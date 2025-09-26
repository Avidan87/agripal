"""
🛠️ AgriPal Utility Functions
Shared helper functions and utilities for the backend.
"""
import asyncio
import logging
import time
import functools
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

from .models import UserSession
from .config import settings

def setup_logging():
    """
    Configure logging for the application
    """
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def measure_execution_time(func):
    """
    Decorator to measure function execution time
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"⏱️ {func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"⏱️ {func.__name__} executed in {execution_time:.2f}ms")
        
        return result
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

async def get_user_session(session_id: Optional[str] = None) -> UserSession:
    """
    Get or create user session
    
    Args:
        session_id: Optional session ID
        
    Returns:
        UserSession object
    """
    if session_id:
        # In a real implementation, this would fetch from database
        # For now, create a basic session
        return UserSession(
            id=session_id,
            user_email="farmer@example.com",
            user_name="Demo Farmer",
            location="Iowa, USA"
        )
    else:
        # Create new session
        return UserSession(
            id=str(uuid.uuid4()),
            user_email="farmer@example.com", 
            user_name="Demo Farmer",
            location="Iowa, USA"
        )

def generate_session_id() -> str:
    """
    Generate a unique session ID
    
    Returns:
        UUID string
    """
    return str(uuid.uuid4())

def validate_image_file(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Validate uploaded image file
    
    Args:
        file_content: File content bytes
        filename: Original filename
        
    Returns:
        Validation result dictionary
    """
    result = {
        "valid": False,
        "error": None,
        "file_size": len(file_content),
        "filename": filename
    }
    
    # Check file size
    if len(file_content) > settings.MAX_FILE_SIZE:
        result["error"] = f"File size {len(file_content)} exceeds maximum {settings.MAX_FILE_SIZE}"
        return result
    
    # Check file extension
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if f'.{file_ext}' not in allowed_extensions:
        result["error"] = f"File type .{file_ext} not allowed. Allowed types: {allowed_extensions}"
        return result
    
    result["valid"] = True
    return result

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    size = float(size_bytes)
    
    while size >= 1024.0 and i < len(size_names) - 1:
        size /= 1024.0
        i += 1
    
    return f"{size:.1f}{size_names[i]}"

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    import re
    # Remove unsafe characters
    safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
    # Add timestamp to ensure uniqueness
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    name, ext = safe_filename.rsplit('.', 1) if '.' in safe_filename else (safe_filename, '')
    
    return f"{name}_{timestamp}.{ext}" if ext else f"{name}_{timestamp}"

class ErrorHandler:
    """
    Centralized error handling utilities
    """
    
    @staticmethod
    def format_error_response(error: Exception, request_id: str = None) -> Dict[str, Any]:
        """
        Format error for API response
        
        Args:
            error: Exception object
            request_id: Optional request ID for tracking
            
        Returns:
            Formatted error dictionary
        """
        return {
            "error": {
                "message": str(error),
                "type": type(error).__name__,
                "request_id": request_id or generate_session_id(),
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    @staticmethod
    def log_error(error: Exception, context: Dict[str, Any] = None):
        """
        Log error with context
        
        Args:
            error: Exception object
            context: Additional context information
        """
        logger = logging.getLogger(__name__)
        context_str = f" Context: {context}" if context else ""
        logger.error(f"❌ {type(error).__name__}: {str(error)}{context_str}")

# Export commonly used functions
__all__ = [
    "setup_logging",
    "measure_execution_time", 
    "get_user_session",
    "generate_session_id",
    "validate_image_file",
    "format_file_size",
    "sanitize_filename",
    "ErrorHandler"
]
