"""
PDF Summarize - Processing Logic Module

This module contains the core processing pipeline logic that orchestrates
language detection, translation, and summarization operations.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import os
import sys
from typing import Callable, List, Union

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.languages_detect import detect_languages


def process_text(
    text: str,
    translator: 'Translator',
    summarize_model: Callable,
    input_max_length: int,
    sum_max_length: int,
    sum_min_length: int,
    num_beams: int,
    support_languages: List[str]
) -> str:
    """
    Process text through the complete summarization pipeline.
    
    This function implements the core logic for multilingual text summarization:
    1. Detect the source language of the input text
    2. Translate to English if necessary (for non-English languages)
    3. Generate summary using the BART model
    4. Translate summary back to original language if needed
    5. Return the final summary in the original language
    
    Args:
        text (str): The input text to process and summarize
        translator (Translator): Translation model instance for language conversion
        summarize_model (Callable): Function to generate summaries using BART
        input_max_length (int): Maximum length for input text processing
        sum_max_length (int): Maximum length of generated summary
        sum_min_length (int): Minimum length of generated summary
        num_beams (int): Number of beams for beam search generation
        support_languages (List[str]): List of supported language codes
        
    Returns:
        str: The final summary text in the original language, or error message
        
    Raises:
        Exception: Various exceptions may be raised during processing steps
        
    Processing Flow:
        - For English text: Direct summarization
        - For supported non-English text: Translate → Summarize → Translate back
        - For unsupported languages: Return error message
        
    Examples:
        >>> # English text (direct processing)
        >>> result = process_text(
        ...     text="English article to summarize...",
        ...     translator=translator_instance,
        ...     summarize_model=bart_model,
        ...     input_max_length=1024,
        ...     sum_max_length=200,
        ...     sum_min_length=20,
        ...     num_beams=2,
        ...     support_languages=['en', 'fr', 'es']
        ... )
        
        >>> # French text (with translation)  
        >>> result = process_text(
        ...     text="Article français à résumer...",
        ...     translator=translator_instance,
        ...     summarize_model=bart_model,
        ...     input_max_length=1024,
        ...     sum_max_length=200,
        ...     sum_min_length=20,
        ...     num_beams=2,
        ...     support_languages=['en', 'fr', 'es']
        ... )
    """
    try:
        # Step 1: Detect the language of input text
        text_lang = detect_languages(text)
        print(f"[INFO] Detected language: {text_lang}")
        
        if text_lang is None:
            return "Unable to detect text language."
        
        # Step 2: Process based on detected language
        if text_lang == "en":
            # Direct processing for English text
            print("[INFO] Processing English text directly")
            summary = summarize_model(
                text=text,
                input_max_length=input_max_length,
                sum_max_length=sum_max_length,
                sum_min_length=sum_min_length,
                num_beams=num_beams,
            )
            return summary
            
        elif text_lang in support_languages:
            # Translation pipeline for supported non-English languages
            print(f"[INFO] Processing {text_lang} text with translation pipeline")
            
            # Step 2a: Translate source text to English
            print("[INFO] Translating to English...")
            translated_text = translator.translate(text, direction="XToEN")
            
            if not translated_text or not translated_text.strip():
                return "Translation to English failed."
            
            print(f"[INFO] Translation successful: {len(translated_text)} characters")
            
            # Step 2b: Generate summary in English
            print("[INFO] Generating summary...")
            english_summary = summarize_model(
                text=translated_text,
                input_max_length=input_max_length,
                sum_max_length=sum_max_length,
                sum_min_length=sum_min_length,
                num_beams=num_beams,
            )
            
            if not english_summary or not english_summary.strip():
                return "Summary generation failed."
            
            # Step 2c: Translate summary back to original language
            print(f"[INFO] Translating summary back to {text_lang}...")
            
            # Prepare text with language prefix for translation model
            prefixed_summary = f">>{text_lang}<< {english_summary}"
            final_summary = translator.translate(prefixed_summary, direction="EnToX")
            
            if not final_summary or not final_summary.strip():
                return "Back-translation failed."
                
            print("[INFO] Processing completed successfully")
            return final_summary
            
        else:
            # Unsupported language
            return f"Language '{text_lang}' is not supported. Supported languages: {', '.join(support_languages)}"
    
    except Exception as e:
        error_msg = f"Error during text processing: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return error_msg


def validate_input_parameters(
    text: str,
    input_max_length: int,
    sum_max_length: int,
    sum_min_length: int,
    num_beams: int
) -> Union[str, None]:
    """
    Validate input parameters for text processing.
    
    Args:
        text (str): Input text to validate
        input_max_length (int): Maximum input length parameter
        sum_max_length (int): Maximum summary length parameter
        sum_min_length (int): Minimum summary length parameter
        num_beams (int): Number of beams parameter
        
    Returns:
        Union[str, None]: Error message if validation fails, None if valid
    """
    if not isinstance(text, str) or not text.strip():
        return "Input text must be a non-empty string."
    
    if input_max_length <= 0:
        return "input_max_length must be positive."
    
    if sum_max_length <= sum_min_length:
        return "sum_max_length must be greater than sum_min_length."
    
    if sum_min_length <= 0:
        return "sum_min_length must be positive."
    
    if num_beams <= 0:
        return "num_beams must be positive."
    
    return None


def get_processing_stats(
    text: str,
    summary: str,
    detected_language: str
) -> dict:
    """
    Generate processing statistics for the summarization task.
    
    Args:
        text (str): Original input text
        summary (str): Generated summary
        detected_language (str): Detected language code
        
    Returns:
        dict: Processing statistics
    """
    return {
        "input_length": len(text),
        "input_words": len(text.split()),
        "summary_length": len(summary),
        "summary_words": len(summary.split()),
        "compression_ratio": len(summary) / len(text) if text else 0,
        "detected_language": detected_language,
        "processing_complete": True
    }
