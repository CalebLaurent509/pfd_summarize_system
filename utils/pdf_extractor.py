"""
PDF Summarize - PDF Text Extraction Module

This module provides functionality to extract text content from PDF files
using PyPDF2. It handles various PDF formats and provides error handling
for corrupted or image-based PDFs.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

import os
import sys
from typing import Optional
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import PyPDF2


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from a PDF file.
    
    This function reads a PDF file and extracts all text content from each page,
    combining them into a single string. It handles various PDF formats and
    provides appropriate error messages for different failure cases.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        
    Returns:
        str: Extracted text content from the PDF, or empty string if extraction fails
        
    Raises:
        None: Function handles all exceptions internally and returns empty string on failure
        
    Examples:
        >>> text = extract_text_from_pdf("document.pdf")
        >>> if text:
        ...     print(f"Extracted {len(text)} characters")
        ... else:
        ...     print("No text could be extracted")
        
        >>> # With full path
        >>> text = extract_text_from_pdf("/home/user/documents/report.pdf")
        
    Note:
        - Works best with text-based PDFs
        - Image-based PDFs (scanned documents) may not extract readable text
        - Encrypted PDFs may require additional handling
        - Large PDFs may take longer to process
    """
    try:
        # Validate input path
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            print(f"[ERROR] PDF file not found: {pdf_path}")
            return ""
        
        if not pdf_file.is_file():
            print(f"[ERROR] Path is not a file: {pdf_path}")
            return ""
        
        if pdf_file.suffix.lower() != '.pdf':
            print(f"[WARNING] File may not be a PDF: {pdf_path}")
        
        print(f"[INFO] Processing PDF: {pdf_file.name}")
        
        # Open and read the PDF file
        with open(pdf_path, "rb") as file:
            try:
                reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    print("[WARNING] PDF is encrypted. Attempting to decrypt...")
                    try:
                        reader.decrypt("")  # Try empty password
                    except Exception as e:
                        print(f"[ERROR] Cannot decrypt PDF: {str(e)}")
                        return ""
                
                total_pages = len(reader.pages)
                print(f"[INFO] PDF contains {total_pages} pages")
                
                if total_pages == 0:
                    print("[WARNING] PDF contains no pages")
                    return ""
                
                # Extract text from all pages
                extracted_text = ""
                successful_pages = 0
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            extracted_text += page_text + "\n"
                            successful_pages += 1
                        else:
                            print(f"[WARNING] No text found on page {page_num}")
                    
                    except Exception as e:
                        print(f"[WARNING] Failed to extract text from page {page_num}: {str(e)}")
                        continue
                
                # Final processing
                extracted_text = extracted_text.strip()
                
                if extracted_text:
                    print(f"[SUCCESS] Extracted text from {successful_pages}/{total_pages} pages")
                    print(f"[INFO] Total characters extracted: {len(extracted_text)}")
                    return extracted_text
                else:
                    print("[WARNING] No text could be extracted from any page")
                    print("[INFO] This may be an image-based PDF requiring OCR")
                    return ""
                    
            except PyPDF2.errors.PdfReadError as e:
                print(f"[ERROR] Cannot read PDF file - corrupted or invalid format: {str(e)}")
                return ""
                
            except Exception as e:
                print(f"[ERROR] Unexpected error while reading PDF: {str(e)}")
                return ""
                
    except PermissionError:
        print(f"[ERROR] Permission denied accessing file: {pdf_path}")
        return ""
        
    except FileNotFoundError:
        print(f"[ERROR] File not found: {pdf_path}")
        return ""
        
    except Exception as e:
        print(f"[ERROR] Unexpected error during PDF processing: {str(e)}")
        return ""


def validate_pdf_file(pdf_path: str) -> tuple[bool, str]:
    """
    Validate if a file is a readable PDF.
    
    Args:
        pdf_path (str): Path to the PDF file to validate
        
    Returns:
        tuple[bool, str]: (is_valid, message) - validation result and message
        
    Examples:
        >>> is_valid, message = validate_pdf_file("document.pdf")
        >>> if is_valid:
        ...     print("PDF is valid and readable")
        ... else:
        ...     print(f"PDF validation failed: {message}")
    """
    try:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            return False, "File does not exist"
        
        if not pdf_file.is_file():
            return False, "Path is not a file"
        
        if pdf_file.suffix.lower() != '.pdf':
            return False, "File does not have .pdf extension"
        
        # Try to open and read basic PDF structure
        with open(pdf_path, "rb") as file:
            try:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                
                if page_count == 0:
                    return False, "PDF contains no pages"
                
                return True, f"Valid PDF with {page_count} pages"
                
            except PyPDF2.errors.PdfReadError:
                return False, "Corrupted or invalid PDF format"
                
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def get_pdf_info(pdf_path: str) -> dict:
    """
    Get detailed information about a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        dict: PDF information including metadata and page count
    """
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            
            info = {
                "file_path": pdf_path,
                "file_size_mb": round(Path(pdf_path).stat().st_size / (1024 * 1024), 2),
                "page_count": len(reader.pages),
                "is_encrypted": reader.is_encrypted,
                "metadata": {},
                "readable": True
            }
            
            # Extract metadata if available
            if reader.metadata:
                for key, value in reader.metadata.items():
                    if isinstance(value, str):
                        info["metadata"][key] = value
            
            return info
            
    except Exception as e:
        return {
            "file_path": pdf_path,
            "error": str(e),
            "readable": False
        }