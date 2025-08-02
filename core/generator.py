"""
PDF Summarize - Main Generator Module

This module provides the main entry point for generating summaries from text input.
It orchestrates the complete pipeline including translation, summarization, and
language detection.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import os
import sys
from typing import Dict, Union, Any
from pathlib import Path

# Add project root to Python path for proper imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.summarizer import summarize_model
from src.translator_models import Translator
from core.logic import process_text
from utils.languages_detect import support_languages, detect_languages
from utils.pdf_extractor import extract_text_from_pdf
from utils.preprocessing import preprocess


def generate_summary(
    text: str,
    input_max_length: int = 1024,
    sum_max_length: int = 200,
    sum_min_length: int = 20,
    num_beams: int = 2
) -> Union[Dict[str, Any], str]:
    """
    Generate a summary for the given text input.
    
    This function implements the complete summarization pipeline:
    1. Initialize translation models
    2. Process text through language detection and translation
    3. Generate summary using BART model
    4. Return results with detected language information
    
    Args:
        text (str): The source text to summarize
        input_max_length (int, optional): Maximum allowed length for input text. 
                                        Defaults to 1024.
        sum_max_length (int, optional): Maximum length of generated summary. 
                                    Defaults to 200.
        sum_min_length (int, optional): Minimum length of generated summary. 
                                    Defaults to 20.
        num_beams (int, optional): Number of beams for beam search generation. 
                                Defaults to 2.
    
    Returns:
        Union[Dict[str, Any], str]: Dictionary containing language and summary text
                                if successful, error message string if failed.
                                
        Success format:
        {
            "lang": str,  # Detected language code
            "text": str   # Generated summary
        }
    
    Raises:
        RuntimeError: If translator initialization fails
        
    Examples:
        >>> text = "Long text to summarize..."
        >>> result = generate_summary(text)
        >>> print(f"Language: {result['lang']}, Summary: {result['text']}")
        
        >>> # Custom parameters
        >>> result = generate_summary(
        ...     text, 
        ...     input_max_length=2048,
        ...     sum_max_length=150
        ... )
    """
    try:
        # Initialize translator with proper model directory path
        model_dir = Path(__file__).parent.parent / "models" / "translation_models"
        translator = Translator(model_dir=str(model_dir))
    except Exception as e:
        raise RuntimeError(f"Failed to initialize translator: {str(e)}") from e

    # Validate and truncate input text if necessary
    if len(text) > input_max_length:
        text = text[:input_max_length]
        print(f"[INFO] Text too long. Using first {input_max_length} characters only.")

    try:
        # Execute the complete processing pipeline
        result = process_text(
            text=text,
            translator=translator,
            summarize_model=summarize_model,
            input_max_length=input_max_length,
            sum_max_length=sum_max_length,
            sum_min_length=sum_min_length,
            num_beams=num_beams,
            support_languages=support_languages
        )
        
        # Detect language of the generated summary
        detected_lang = detect_languages(result)
        
        return {
            "lang": detected_lang,
            "text": result
        }
        
    except Exception as e:
        return f"[ERROR] An error occurred during processing: {str(e)}"


def main(file_path: str) -> None:
    """
    Main function for command-line usage.
    
    Processes a sample PDF file located in the same directory as this script.
    This function demonstrates the complete PDF-to-summary pipeline.
    
    Raises:
        FileNotFoundError: If the sample PDF file doesn't exist
    """
    # Define path to sample PDF file
    pdf_path = Path(__file__).parent / file_path
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"Sample PDF file '{pdf_path}' not found.")
    
    try:
        # Extract and preprocess text from PDF
        print(f"[INFO] Processing PDF: {pdf_path}")
        raw_text = extract_text_from_pdf(str(pdf_path))
        
        if not raw_text.strip():
            print("[WARNING] No text extracted from PDF.")
            return
            
        processed_text = preprocess(raw_text)
        print(f"[INFO] Extracted {len(processed_text)} characters after preprocessing.")
        
        # Generate summary with extended input length for PDFs
        result = generate_summary(
            processed_text,
            input_max_length=4024,  # Larger limit for PDF content
            sum_max_length=200,
            sum_min_length=20,
            num_beams=2
        )
        
        # Display results
        if isinstance(result, dict):
            print(f"\n[SUCCESS] Summary generated:")
            print(f"Language: {result['lang']}")
            print(f"Summary: {result['text']}")
        else:
            print(f"\n[ERROR] {result}")
            
    except Exception as e:
        print(f"[ERROR] Failed to process PDF: {str(e)}")


if __name__ == "__main__":
    main("sample.pdf")