"""
PDF Summarize - Language Detection Module

This module provides language detection functionality using the langdetect library.
It supports detection of multiple Romance languages and validates against 
a predefined list of supported languages.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import os
import sys
from typing import Optional, List

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langdetect import detect, DetectorFactory, LangDetectException


# List of supported language codes
# These are the languages supported by the translation models
SUPPORT_LANGUAGES: List[str] = ['en', 'ca', 'es', 'fr', 'it', 'pt', 'ro']

# Set random seed for reproducible language detection results
DetectorFactory.seed = 0


def detect_languages(text: str) -> Optional[str]:
    """
    Detect the language of the given text and validate against supported languages.
    
    This function uses the langdetect library to identify the language of input text
    and checks if it's in the list of supported languages for the summarization system.
    
    Args:
        text (str): The input text to analyze for language detection
        
    Returns:
        Optional[str]: The detected language code if supported, None otherwise
                        Examples: 'en', 'fr', 'es', 'it', 'pt', 'ro', 'ca'
                    
    Raises:
        None: Function handles all exceptions internally and returns None on failure
        
    Examples:
        >>> detect_languages("Hello, how are you?")
        'en'
        
        >>> detect_languages("Bonjour, comment allez-vous?")
        'fr'
        
        >>> detect_languages("Hola, ¿cómo estás?")
        'es'
        
        >>> detect_languages("Very short text")  # May fail detection
        None
        
        >>> detect_languages("Text in unsupported language")
        None
        
    Note:
        - Requires sufficient text length for accurate detection
        - Short texts may result in detection failure
        - Only returns languages supported by the translation models
        - Results are deterministic due to fixed random seed
    """
    try:
        # Validate input
        if not isinstance(text, str) or not text.strip():
            print("[WARNING] Empty or invalid text provided for language detection.")
            return None
        
        # Perform language detection
        detected_lang = detect(text)
        
        # Check if detected language is supported
        if detected_lang not in SUPPORT_LANGUAGES:
            print(f"[WARNING] Detected language '{detected_lang}' is not supported.")
            print(f"[INFO] Supported languages: {', '.join(SUPPORT_LANGUAGES)}")
            return None
        
        print(f"[INFO] Language detected: {detected_lang}")
        return detected_lang
        
    except LangDetectException as e:
        print(f"[ERROR] Unable to detect text language: {str(e)}")
        print("[INFO] This may occur with very short or mixed-language text.")
        return None
    
    except Exception as e:
        print(f"[ERROR] Unexpected error during language detection: {str(e)}")
        return None


def is_language_supported(lang_code: str) -> bool:
    """
    Check if a language code is supported by the system.
    
    Args:
        lang_code (str): Language code to check (e.g., 'en', 'fr', 'es')
        
    Returns:
        bool: True if language is supported, False otherwise
        
    Examples:
        >>> is_language_supported('en')
        True
        
        >>> is_language_supported('zh')
        False
    """
    return lang_code in SUPPORT_LANGUAGES


def get_supported_languages() -> List[str]:
    """
    Get the list of supported language codes.
    
    Returns:
        List[str]: List of supported language codes
        
    Examples:
        >>> get_supported_languages()
        ['en', 'ca', 'es', 'fr', 'it', 'pt', 'ro']
    """
    return SUPPORT_LANGUAGES.copy()


def get_language_info() -> dict:
    """
    Get detailed information about supported languages.
    
    Returns:
        dict: Dictionary mapping language codes to language names
    """
    language_names = {
        'en': 'English',
        'ca': 'Catalan', 
        'es': 'Spanish',
        'fr': 'French',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ro': 'Romanian'
    }
    
    return {
        'supported_codes': SUPPORT_LANGUAGES,
        'language_names': language_names,
        'total_supported': len(SUPPORT_LANGUAGES),
        'detection_library': 'langdetect',
        'deterministic_results': True
    }


# Export the support_languages for backward compatibility
support_languages = SUPPORT_LANGUAGES