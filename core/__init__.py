"""
PDF Summarize - Core Module

This module contains the core functionality for PDF text summarization,
including the main generation logic, processing pipeline, and BART model implementation.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .generator import generate_summary
from .summarizer import summarize_model

__all__ = [
    "generate_summary",
    "summarize_model",
]
