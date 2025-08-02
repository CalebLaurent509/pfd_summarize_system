"""
PDF Summarize - Text Preprocessing Module

This module provides comprehensive text preprocessing functionality including
normalization, cleaning, tokenization, stop word removal, and lemmatization
using NLTK (Natural Language Toolkit).

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import re
import string
from typing import List, Optional
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Set custom NLTK data path for model storage
NLTK_DATA_PATH = Path(__file__).parent.parent / "models" / "nltk_data"
NLTK_DATA_PATH.mkdir(parents=True, exist_ok=True)
nltk.data.path.insert(0, str(NLTK_DATA_PATH))


def download_nltk_dependencies() -> bool:
    """
    Download required NLTK data if not already present.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
    """
    required_data = [
        ('stopwords', 'corpora/stopwords'),
        ('punkt_tab', 'tokenizers/punkt_tab'),
        ('wordnet', 'corpora/wordnet')
    ]
    
    success = True
    
    for name, path in required_data:
        try:
            nltk.data.find(path)
            print(f"[INFO] NLTK data '{name}' already available")
        except LookupError:
            try:
                print(f"[INFO] Downloading NLTK data: {name}")
                nltk.download(name, download_dir=str(NLTK_DATA_PATH), quiet=True)
            except Exception as e:
                print(f"[ERROR] Failed to download NLTK data '{name}': {str(e)}")
                success = False
    
    return success


def preprocess(
    text: str,
    remove_stopwords: bool = True,
    apply_lemmatization: bool = True,
    remove_punctuation: bool = True,
    remove_digits: bool = True,
    to_lowercase: bool = True,
    language: str = 'english'
) -> str:
    """
    Perform comprehensive text preprocessing for NLP tasks.
    
    This function applies multiple preprocessing steps to clean and normalize text:
    1. Unicode normalization and special character handling
    2. Whitespace normalization
    3. Case conversion (optional)
    4. Punctuation removal (optional)
    5. Digit removal (optional)
    6. Tokenization
    7. Stop word removal (optional)
    8. Lemmatization (optional)
    9. Text reconstruction
    
    Args:
        text (str): Input text to preprocess
        remove_stopwords (bool, optional): Remove common stop words. Defaults to True.
        apply_lemmatization (bool, optional): Apply word lemmatization. Defaults to True.
        remove_punctuation (bool, optional): Remove punctuation marks. Defaults to True.
        remove_digits (bool, optional): Remove numeric digits. Defaults to True.
        to_lowercase (bool, optional): Convert to lowercase. Defaults to True.
        language (str, optional): Language for stop words. Defaults to 'english'.
        
    Returns:
        str: Preprocessed and cleaned text
        
    Raises:
        ValueError: If input text is empty or invalid
        RuntimeError: If NLTK dependencies are missing
        
    Examples:
        >>> text = "The quick brown fox jumps over the lazy dog!"
        >>> cleaned = preprocess(text)
        >>> print(cleaned)
        "quick brown fox jump lazy dog"
        
        >>> # Custom preprocessing
        >>> cleaned = preprocess(
        ...     text,
        ...     remove_stopwords=False,
        ...     remove_punctuation=False
        ... )
        >>> print(cleaned)
        "the quick brown fox jumps over the lazy dog!"
        
        >>> # PDF text preprocessing
        >>> pdf_text = "PDF Bookmark Sample Page 1 of 4  \\xa0 \\xa0 The text..."
        >>> cleaned = preprocess(pdf_text)
    """
    # Input validation
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    if not text.strip():
        return ""
    
    # Ensure NLTK dependencies are available
    if not download_nltk_dependencies():
        raise RuntimeError("Failed to download required NLTK data")
    
    try:
        # Step 1: Unicode normalization and special character handling
        text = text.replace('\xa0', ' ')  # Replace non-breaking spaces
        text = text.replace('\u00a0', ' ')  # Another non-breaking space variant
        text = text.replace('\n', ' ')     # Replace newlines with spaces
        text = text.replace('\t', ' ')     # Replace tabs with spaces
        text = text.replace('\r', ' ')     # Replace carriage returns
        
        # Step 2: Normalize whitespace (multiple spaces -> single space)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 3: Convert to lowercase if requested
        if to_lowercase:
            text = text.lower()
        
        # Step 4: Remove punctuation if requested
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Step 5: Remove digits if requested
        if remove_digits:
            text = re.sub(r'\d+', '', text)
        
        # Step 6: Tokenization
        try:
            words = word_tokenize(text)
        except Exception as e:
            print(f"[WARNING] Tokenization failed, using simple split: {str(e)}")
            words = text.split()
        
        # Step 7: Remove stop words if requested
        if remove_stopwords:
            try:
                stop_words = set(stopwords.words(language))
                words = [word for word in words if word and word not in stop_words]
            except Exception as e:
                print(f"[WARNING] Stop words removal failed: {str(e)}")
        
        # Step 8: Apply lemmatization if requested
        if apply_lemmatization:
            try:
                lemmatizer = WordNetLemmatizer()
                words = [lemmatizer.lemmatize(word) for word in words if word]
            except Exception as e:
                print(f"[WARNING] Lemmatization failed: {str(e)}")
        
        # Step 9: Reconstruct cleaned text
        cleaned_text = ' '.join(word for word in words if word.strip())
        
        # Final whitespace cleanup
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {str(e)}")
        return text  # Return original text if preprocessing fails


def preprocess_for_summarization(text: str) -> str:
    """
    Apply preprocessing optimized for summarization tasks.
    
    This function applies a specific preprocessing configuration that works
    well for text summarization, preserving important structural information
    while cleaning noise.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Text preprocessed for summarization
        
    Examples:
        >>> text = "This is a sample document with various issues!"
        >>> clean_text = preprocess_for_summarization(text)
    """
    return preprocess(
        text,
        remove_stopwords=False,    # Keep stop words for summarization
        apply_lemmatization=True,  # Apply lemmatization
        remove_punctuation=False,  # Keep punctuation for structure
        remove_digits=False,       # Keep numbers (may be important)
        to_lowercase=False,        # Preserve case information
        language='english'
    )


def preprocess_for_translation(text: str) -> str:
    """
    Apply preprocessing optimized for translation tasks.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Text preprocessed for translation
    """
    return preprocess(
        text,
        remove_stopwords=False,    # Keep all words for translation
        apply_lemmatization=False, # Don't lemmatize for translation
        remove_punctuation=False,  # Keep punctuation
        remove_digits=False,       # Keep numbers
        to_lowercase=False,        # Preserve case
        language='english'
    )


def get_text_statistics(text: str) -> dict:
    """
    Generate statistics about the input text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        dict: Text statistics including character, word, and sentence counts
    """
    if not text:
        return {
            "characters": 0,
            "words": 0,
            "sentences": 0,
            "average_word_length": 0
        }
    
    # Basic counts
    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len(re.split(r'[.!?]+', text))
    
    # Average word length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    return {
        "characters": char_count,
        "words": word_count,
        "sentences": sentence_count,
        "average_word_length": round(avg_word_length, 2)
    }


# Initialize NLTK dependencies on module import
try:
    download_nltk_dependencies()
except Exception as e:
    print(f"[WARNING] Failed to initialize NLTK dependencies: {e}")


# Example usage and testing
if __name__ == "__main__":
    # Example text with various formatting issues
    sample_text = (
        "PDF Bookmark Sample Page 1 of 4  \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 \xa0 "
        "The left pane displays the available bookmarks for this PDF. "
        "You may need to enable the display of bookmarks in Adobe Acrobat Reader."
    )
    
    print("Original text:")
    print(repr(sample_text))
    print("\nCleaned text:")
    cleaned = preprocess(sample_text)
    print(cleaned)
    
    print("\nText statistics:")
    stats = get_text_statistics(sample_text)
    for key, value in stats.items():
        print(f"{key}: {value}")