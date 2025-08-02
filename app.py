"""
PDF Summarize - Flask Web Application

This module provides a Flask web API for PDF text summarization with translation
support. It offers RESTful endpoints for processing PDF documents and generating
summaries in multiple languages.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.exceptions import BadRequest, InternalServerError
import traceback

from core.generator import generate_summary


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])  # Configure CORS for frontend

# Application configuration
app.config.update(
    JSON_SORT_KEYS=False,
    JSONIFY_PRETTYPRINT_REGULAR=True,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
)


@app.errorhandler(400)
def bad_request(error) -> Response:
    """Handle 400 Bad Request errors."""
    return jsonify({
        "error": "Bad Request",
        "message": "Invalid request format or missing required parameters",
        "status_code": 400
    }), 400


@app.errorhandler(500)
def internal_error(error) -> Response:
    """Handle 500 Internal Server errors."""
    return jsonify({
        "error": "Internal Server Error",
        "message": "An unexpected error occurred while processing your request",
        "status_code": 500
    }), 500


@app.route('/', methods=['GET'])
def health_check() -> Response:
    """
    Health check endpoint to verify API status.
    
    Returns:
        Response: JSON response with API status and version information
        
    Examples:
        GET / -> {"status": "healthy", "version": "1.0.0", "service": "PDF Summarizer API"}
    """
    return jsonify({
        "status": "healthy",
        "service": "PDF Summarizer API",
        "version": "1.0.0",
        "endpoints": {
            "/": "Health check",
            "/summarize": "POST - Generate text summary",
            "/api/info": "GET - API information"
        }
    })


@app.route('/api/info', methods=['GET'])
def api_info() -> Response:
    """
    Get API information and capabilities.
    
    Returns:
        Response: JSON response with API capabilities and supported features
    """
    return jsonify({
        "service": "PDF Summarizer API",
        "version": "1.0.0",
        "description": "REST API for PDF text summarization with multilingual support",
        "capabilities": {
            "summarization": {
                "model": "facebook/bart-large-cnn",
                "max_input_length": 1024,
                "max_output_length": 150,
                "supported_languages": ["en", "fr", "es", "it", "pt", "ro", "ca"]
            },
            "translation": {
                "models": ["Helsinki-NLP/opus-mt-ROMANCE-en", "Helsinki-NLP/opus-mt-en-ROMANCE"],
                "supported_pairs": ["Romance languages <-> English"]
            },
            "file_processing": {
                "supported_formats": ["PDF", "plain text"],
                "max_file_size": "16MB"
            }
        },
        "endpoints": {
            "POST /summarize": {
                "description": "Generate summary from text",
                "parameters": {
                    "text": "string (required) - Text to summarize",
                    "max_length": "integer (optional) - Maximum summary length (default: 150)",
                    "min_length": "integer (optional) - Minimum summary length (default: 50)",
                    "language": "string (optional) - Target language for summary (default: auto-detect)"
                }
            }
        }
    })


@app.route('/summarize', methods=['POST'])
def summarize() -> Response:
    """
    Generate a summary from provided text.
    
    This endpoint accepts text input and returns a generated summary using
    the BART model. It supports various parameters for customizing the
    summarization process.
    
    Request Body:
        {
            "text": "Text to summarize (required)",
            "max_length": 150,  // Optional: Maximum summary length
            "min_length": 50,   // Optional: Minimum summary length  
            "language": "en"    // Optional: Target language
        }
        
    Returns:
        Response: JSON response with generated summary and metadata
        
    Response Format:
        {
            "summary": "Generated summary text",
            "metadata": {
                "original_length": 1000,
                "summary_length": 120,
                "compression_ratio": 0.12,
                "processing_time": 2.5,
                "model_used": "facebook/bart-large-cnn"
            },
            "status": "success"
        }
        
    Raises:
        400: If request is malformed or missing required text
        500: If summarization process fails
        
    Examples:
        >>> # Basic summarization
        >>> curl -X POST http://localhost:5000/summarize \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{"text": "Long document text here..."}'
        
        >>> # Custom parameters
        >>> curl -X POST http://localhost:5000/summarize \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{
        ...          "text": "Long document text...",
        ...          "max_length": 200,
        ...          "min_length": 75
        ...      }'
    """
    try:
        # Validate request content type
        if not request.is_json:
            logger.warning("Invalid content type: %s", request.content_type)
            return jsonify({
                "error": "Invalid Content-Type",
                "message": "Request must have Content-Type: application/json",
                "status_code": 400
            }), 400
        
        # Parse request data
        data = request.get_json()
        if not data:
            logger.warning("Empty request body received")
            return jsonify({
                "error": "Empty Request",
                "message": "Request body cannot be empty",
                "status_code": 400
            }), 400
        
        # Validate required text parameter
        text = data.get('text')
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Missing or invalid text parameter")
            return jsonify({
                "error": "Missing Text",
                "message": "The 'text' parameter is required and must be a non-empty string",
                "status_code": 400
            }), 400
        
        # Extract optional parameters with defaults
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 50)
        target_language = data.get('language', 'auto')
        
        # Validate parameters
        if not isinstance(max_length, int) or max_length < 10 or max_length > 1000:
            return jsonify({
                "error": "Invalid max_length",
                "message": "max_length must be an integer between 10 and 1000",
                "status_code": 400
            }), 400
        
        if not isinstance(min_length, int) or min_length < 5 or min_length >= max_length:
            return jsonify({
                "error": "Invalid min_length", 
                "message": f"min_length must be an integer between 5 and {max_length-1}",
                "status_code": 400
            }), 400
        
        # Log request details
        logger.info("Summarization request received - text length: %d, max_length: %d, min_length: %d", 
                    len(text), max_length, min_length)
        
        # Generate summary using core generator
        import time
        start_time = time.time()
        
        result = generate_summary(
            text,
            input_max_length=1024,  # Default input length
            sum_max_length=max_length,
            sum_min_length=min_length,
            num_beams=2  # Default beam search parameter
        )
        
        processing_time = time.time() - start_time
        
        # Handle different result formats from generator
        if isinstance(result, dict):
            summary = result.get('text') or result.get('summary', '')
            detected_language = result.get('lang', 'unknown')
            if not summary:
                logger.error("Generator returned empty summary in dict format")
                return jsonify({
                    "error": "Empty Summary",
                    "message": "Summary generation produced no output",
                    "status_code": 500
                }), 500
        elif isinstance(result, str):
            summary = result
            detected_language = 'auto'
        else:
            logger.error("Generator returned unexpected format: %s", type(result))
            return jsonify({
                "error": "Invalid Response Format",
                "message": "Summary generator returned unexpected format",
                "status_code": 500
            }), 500
        
        # Calculate metrics
        original_length = len(text)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        
        # Prepare response
        response_data = {
            "summary": summary.strip(),
            "language": detected_language,
            "metadata": {
                "original_length": original_length,
                "summary_length": summary_length,
                "compression_ratio": round(compression_ratio, 3),
                "processing_time": round(processing_time, 2),
                "model_used": "facebook/bart-large-cnn",
                "parameters": {
                    "max_length": max_length,
                    "min_length": min_length,
                    "detected_language": detected_language
                }
            },
            "status": "success"
        }
        
        logger.info("Summarization completed successfully - compression ratio: %.3f, time: %.2fs", 
                    compression_ratio, processing_time)
        
        return jsonify(response_data)
        
    except Exception as e:
        # Log full error details
        logger.error("Summarization failed: %s", str(e))
        logger.error("Full traceback: %s", traceback.format_exc())
        
        return jsonify({
            "error": "Summarization Failed",
            "message": f"An error occurred during text summarization: {str(e)}",
            "status_code": 500
        }), 500


@app.route('/api/validate', methods=['POST'])
def validate_text() -> Response:
    """
    Validate text input without generating summary.
    
    Returns:
        Response: JSON response with text validation results
    """
    try:
        data = request.get_json()
        text = data.get('text', '') if data else ''
        
        validation_result = {
            "valid": bool(text and text.strip()),
            "length": len(text) if text else 0,
            "word_count": len(text.split()) if text else 0,
            "estimated_processing_time": len(text) / 1000 if text else 0  # Rough estimate
        }
        
        return jsonify(validation_result)
        
    except Exception as e:
        logger.error("Text validation failed: %s", str(e))
        return jsonify({
            "error": "Validation Failed",
            "message": str(e),
            "status_code": 500
        }), 500


# Development server configuration
if __name__ == '__main__':
    # Set up development environment
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info("Starting PDF Summarizer API server...")
    logger.info("Debug mode: %s", debug_mode)
    logger.info("Port: %d", port)
    
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug_mode,
            threaded=True
        )
    except Exception as e:
        logger.error("Failed to start server: %s", str(e))
        sys.exit(1)