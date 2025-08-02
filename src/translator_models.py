"""
PDF Summarize - Translation Models Module

This module provides translation functionality using Hugging Face transformers
with Helsinki-NLP MarianMT models. It supports translation between multiple
languages for PDF document processing and summarization.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import os
import sys
from typing import Dict, Optional, List, Tuple, Union
from pathlib import Path
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import MarianMTModel, MarianTokenizer


# Configure logging
logger = logging.getLogger(__name__)


class Translator:
    """
    Professional translation class for Romance languages and English using MarianMT models.
    
    This class provides efficient translation capabilities between Romance languages
    (Romanian, Spanish, French, Italian, Portuguese, Catalan) and English using
    pre-trained Helsinki-NLP MarianMT models with local caching support.
    
    Attributes:
        model_dir (str): Directory for saving/loading models locally
        XToEn_model (MarianMTModel): Model for translating Romance languages to English
        XToEn_tokenizer (MarianTokenizer): Tokenizer for XToEn_model
        EnToX_model (MarianMTModel): Model for translating English to Romance languages
        EnToX_tokenizer (MarianTokenizer): Tokenizer for EnToX_model
        supported_languages (List[str]): List of supported language codes
        
    Examples:
        >>> translator = Translator("./models")
        >>> # Translate French to English
        >>> result = translator.translate("Bonjour le monde", direction="XToEN")
        >>> print(result)
        "Hello world"
        
        >>> # Translate English to French (targeting French)
        >>> result = translator.translate("Hello world", direction="EnToX")
        >>> # Note: EnToX model may output various Romance languages
        
        >>> # Batch translation
        >>> texts = ["Bonjour", "Au revoir"]
        >>> results = translator.translate(texts, direction="XToEN")
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize the Translator with model directory for local caching.
        
        Args:
            model_dir (str): Directory path for saving/loading translation models
            
        Raises:
            OSError: If model directory cannot be created or accessed
            RuntimeError: If model loading fails
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model identifiers from Hugging Face
        self.XToEN_id = "Helsinki-NLP/opus-mt-ROMANCE-en"
        self.EnToX_id = "Helsinki-NLP/opus-mt-en-ROMANCE"
        
        # Supported Romance languages
        self.supported_languages = ['ro', 'es', 'fr', 'it', 'pt', 'ca', 'en']
        
        # Define local model paths
        self.x_to_en_path = self.model_dir / "XToEN"
        self.en_to_x_path = self.model_dir / "EnToX"
        
        # Initialize models
        self._load_models()
        
        logger.info("Translator initialized with model directory: %s", model_dir)
    
    def _load_models(self) -> None:
        """
        Load or download translation models for both directions.
        
        This method handles the downloading and caching of MarianMT models
        for efficient local usage and faster subsequent loads.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Load X -> EN model (Romance languages to English)
            if not self.x_to_en_path.exists():
                logger.info("Downloading Romance -> English translation model...")
                self.XToEn_model = MarianMTModel.from_pretrained(self.XToEN_id)
                self.XToEn_tokenizer = MarianTokenizer.from_pretrained(self.XToEN_id)
                
                # Save models locally
                self.XToEn_model.save_pretrained(str(self.x_to_en_path))
                self.XToEn_tokenizer.save_pretrained(str(self.x_to_en_path))
                logger.info("Romance -> English model saved to: %s", self.x_to_en_path)
            else:
                logger.info("Loading cached Romance -> English model...")
                self.XToEn_model = MarianMTModel.from_pretrained(str(self.x_to_en_path))
                self.XToEn_tokenizer = MarianTokenizer.from_pretrained(str(self.x_to_en_path))
            
            # Load EN -> X model (English to Romance languages)
            if not self.en_to_x_path.exists():
                logger.info("Downloading English -> Romance translation model...")
                self.EnToX_model = MarianMTModel.from_pretrained(self.EnToX_id)
                self.EnToX_tokenizer = MarianTokenizer.from_pretrained(self.EnToX_id)
                
                # Save models locally
                self.EnToX_model.save_pretrained(str(self.en_to_x_path))
                self.EnToX_tokenizer.save_pretrained(str(self.en_to_x_path))
                logger.info("English -> Romance model saved to: %s", self.en_to_x_path)
            else:
                logger.info("Loading cached English -> Romance model...")
                self.EnToX_model = MarianMTModel.from_pretrained(str(self.en_to_x_path))
                self.EnToX_tokenizer = MarianTokenizer.from_pretrained(str(self.en_to_x_path))
            
            logger.info("All translation models loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load translation models: %s", str(e))
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def translate(
        self, 
        text: Union[str, List[str]], 
        direction: str = "XToEN",
        max_length: int = 512,
        num_beams: int = 4
    ) -> Union[str, List[str]]:
        """
        Translate text in the specified direction.
        
        Args:
            text (str or List[str]): Text or list of texts to translate
            direction (str): Translation direction ("XToEN" or "EnToX")
                - "XToEN": Romance languages to English
                - "EnToX": English to Romance languages
            max_length (int, optional): Maximum length of output. Defaults to 512.
            num_beams (int, optional): Number of beams for beam search. Defaults to 4.
        
        Returns:
            str or List[str]: Translated text(s) in the same format as input
            
        Raises:
            ValueError: If direction is invalid or text is empty
            RuntimeError: If translation fails
            
        Examples:
            >>> translator = Translator("./models")
            >>> # Single text translation
            >>> result = translator.translate("Bonjour le monde", "XToEN")
            >>> print(result)
            "Hello world"
            
            >>> # Batch translation
            >>> texts = ["Hola", "Gracias", "Adiós"]
            >>> results = translator.translate(texts, "XToEN")
            >>> print(results)
            ["Hello", "Thank you", "Goodbye"]
            
            >>> # English to Romance (may output various Romance languages)
            >>> result = translator.translate("Hello", "EnToX")
        """
        # Input validation
        if not text:
            raise ValueError("Text cannot be empty")
        
        if direction not in ["XToEN", "EnToX"]:
            raise ValueError("Invalid direction. Use 'XToEN' or 'EnToX'")
        
        # Select appropriate model and tokenizer
        if direction == "XToEN":
            model = self.XToEn_model
            tokenizer = self.XToEn_tokenizer
        else:  # EnToX
            model = self.EnToX_model
            tokenizer = self.EnToX_tokenizer
        
        # Handle single string input
        single_input = isinstance(text, str)
        if single_input:
            text = [text]
        
        try:
            # Tokenize input texts
            tokenized = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            )
            
            # Generate translations
            with logger_context("Translating texts"):
                translated = model.generate(
                    **tokenized,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode translations
            translated_texts = [
                tokenizer.decode(t, skip_special_tokens=True) 
                for t in translated
            ]
            
            # Return single string if input was single string
            if single_input:
                return translated_texts[0]
            
            return translated_texts
            
        except Exception as e:
            logger.error("Translation failed: %s", str(e))
            raise RuntimeError(f"Translation failed: {str(e)}")
    
    def translate_to_english(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Convenience method to translate Romance languages to English.
        
        Args:
            text (str or List[str]): Text in Romance language(s) to translate
            
        Returns:
            str or List[str]: English translation(s)
        """
        return self.translate(text, direction="XToEN")
    
    def translate_from_english(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Convenience method to translate English to Romance languages.
        
        Note: The output language is determined by the model and may vary.
        
        Args:
            text (str or List[str]): English text to translate
            
        Returns:
            str or List[str]: Romance language translation(s)
        """
        return self.translate(text, direction="EnToX")
    
    def is_language_supported(self, language_code: str) -> bool:
        """
        Check if a language is supported for translation.
        
        Args:
            language_code (str): Language code to check (e.g., 'fr', 'es', 'en')
            
        Returns:
            bool: True if language is supported, False otherwise
        """
        return language_code.lower() in self.supported_languages
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List[str]: List of supported language codes
        """
        return self.supported_languages.copy()
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about loaded models.
        
        Returns:
            Dict[str, str]: Model information including paths and IDs
        """
        return {
            "model_dir": str(self.model_dir),
            "x_to_en_model": self.XToEN_id,
            "en_to_x_model": self.EnToX_id,
            "x_to_en_path": str(self.x_to_en_path),
            "en_to_x_path": str(self.en_to_x_path),
            "supported_languages": ", ".join(self.supported_languages)
        }


class logger_context:
    """Simple context manager for logging translation operations."""
    
    def __init__(self, operation: str):
        self.operation = operation
    
    def __enter__(self):
        logger.debug("Starting: %s", self.operation)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            logger.debug("Completed: %s", self.operation)
        else:
            logger.error("Failed: %s - %s", self.operation, str(exc_val))


# Example usage and testing
if __name__ == "__main__":
    # Test the translator
    try:
        print("Initializing translator...")
        translator = Translator("./models")
        
        # Test translations
        print("\nTesting Romance -> English translations:")
        test_texts = {
            "fr": "Bonjour le monde",
            "es": "Hola mundo", 
            "it": "Ciao mondo",
            "pt": "Olá mundo"
        }
        
        for lang, text in test_texts.items():
            try:
                result = translator.translate_to_english(text)
                print(f"{lang.upper()}: {text} -> EN: {result}")
            except Exception as e:
                print(f"Translation failed for {lang}: {e}")
        
        # Test batch translation
        print("\nTesting batch translation:")
        batch_texts = ["Bonjour", "Merci", "Au revoir"]
        results = translator.translate_to_english(batch_texts)
        for original, translated in zip(batch_texts, results):
            print(f"FR: {original} -> EN: {translated}")
        
        # Model info
        print("\nModel information:")
        info = translator.get_model_info()
        for key, value in info.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Translator test failed: {e}")
