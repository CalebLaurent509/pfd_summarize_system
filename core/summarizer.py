"""
PDF Summarize - BART Summarization Module

This module implements the BART (Bidirectional and Auto-Regressive Transformers)
model for text summarization using Hugging Face Transformers.

The module handles model loading, caching, and text summarization with 
configurable parameters for summary length and quality.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, BartForConditionalGeneration


# Global model and tokenizer instances for reuse
_model = None
_tokenizer = None


def _load_model() -> tuple[BartForConditionalGeneration, AutoTokenizer]:
    """
    Load and cache the BART model and tokenizer.
    
    This function loads the Facebook BART-large-CNN model for summarization
    and caches it locally to avoid repeated downloads.
    
    Returns:
        tuple: (model, tokenizer) - The loaded BART model and its tokenizer
        
    Raises:
        RuntimeError: If model loading fails
    """
    global _model, _tokenizer
    
    if _model is None or _tokenizer is None:
        try:
            # Define local directory for model caching
            local_dir = Path(__file__).parent.parent / "models" / "Bart"
            local_dir.mkdir(parents=True, exist_ok=True)
            
            model_name = "facebook/bart-large-cnn"
            
            # Load model and tokenizer from Hugging Face
            print(f"[INFO] Loading BART model: {model_name}")
            _model = BartForConditionalGeneration.from_pretrained(model_name)
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Cache models locally for future use
            _model.save_pretrained(local_dir)
            _tokenizer.save_pretrained(local_dir)
            print(f"[INFO] Models cached to: {local_dir}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load BART model: {str(e)}") from e
    
    return _model, _tokenizer


def summarize_model(
    text: str,
    input_max_length: int,
    sum_max_length: int,
    sum_min_length: int,
    num_beams: int
) -> str:
    """
    Generate a summary for the given text using BART model.
    
    This function tokenizes the input text, generates a summary using beam search,
    and returns the decoded summary text.
    
    Args:
        text (str): The input text to summarize
        input_max_length (int): Maximum length for input tokenization
        sum_max_length (int): Maximum length of the generated summary
        sum_min_length (int): Minimum length of the generated summary  
        num_beams (int): Number of beams for beam search decoding
        
    Returns:
        str: The generated summary text
        
    Raises:
        ValueError: If input text is empty or invalid
        RuntimeError: If summarization process fails
        
    Examples:
        >>> text = "Long article text here..."
        >>> summary = summarize_model(
        ...     text=text,
        ...     input_max_length=1024,
        ...     sum_max_length=150,
        ...     sum_min_length=50,
        ...     num_beams=4
        ... )
        >>> print(summary)
        "Generated summary of the article..."
    """
    # Input validation
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string.")
    
    # Load model and tokenizer
    model, tokenizer = _load_model()
    
    try:
        # Tokenize input text with proper truncation
        input_tokens = tokenizer(
            [text],
            max_length=input_max_length,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        
        print(f"[INFO] Input tokenized to {input_tokens['input_ids'].shape[1]} tokens")
        
        # Generate summary using beam search
        summary_ids = model.generate(
            input_tokens["input_ids"],
            attention_mask=input_tokens.get("attention_mask"),
            max_length=sum_max_length,
            min_length=sum_min_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,  # Prevent repetition
            do_sample=False  # Use deterministic generation
        )
        
        # Decode the generated summary
        summary = tokenizer.batch_decode(
            summary_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"[INFO] Summary generated: {len(summary)} characters")
        return summary.strip()
        
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}") from e


def get_model_info() -> dict:
    """
    Get information about the loaded BART model.
    
    Returns:
        dict: Model information including name, parameters, and status
    """
    model, tokenizer = _load_model()
    
    return {
        "model_name": "facebook/bart-large-cnn",
        "model_type": "BartForConditionalGeneration",
        "tokenizer_type": "BartTokenizer",
        "vocab_size": tokenizer.vocab_size,
        "max_position_embeddings": model.config.max_position_embeddings,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "is_loaded": _model is not None and _tokenizer is not None
    }


# Initialize model on module import for faster subsequent calls
try:
    _load_model()
except (ImportError, RuntimeError, OSError) as e:
    print(f"[WARNING] Failed to preload BART model: {e}")
    print("[INFO] Model will be loaded on first use")