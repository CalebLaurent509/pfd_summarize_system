#!/bin/bash
"""
PDF Summarizer - Quick Start Script

This script sets up and starts the PDF summarization application with
one command for easy deployment and testing.

Author: Caleb Laurent
Date: 2025
License: MIT
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Main setup function
main() {
    print_header "PDF Summarizer - Quick Start Setup"
    print_header "====================================="
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        exit 1
    fi
    
    print_status "Python 3 found: $(python3 --version)"
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip."
        exit 1
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            print_error "Failed to create virtual environment"
            exit 1
        fi
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing dependencies (this may take several minutes)..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        print_error "Failed to install dependencies"
        exit 1
    fi
    
    # Download NLTK data
    print_status "Downloading NLTK data..."
    python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('wordnet', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Warning: NLTK download failed: {e}')
"
    
    # Validate project
    print_status "Validating project setup..."
    python3 validate_project.py
    
    if [ $? -eq 0 ]; then
        print_header ""
        print_header "SETUP COMPLETED SUCCESSFULLY!"
        print_header "=============================="
        print_status "The PDF Summarizer is ready to use!"
        print_status ""
        print_status "Next steps:"
        print_status "1. Activate the virtual environment: source venv/bin/activate"
        print_status "2. Start the server: python3 run.py"
        print_status "3. Open your browser to: http://localhost:5000"
        print_status ""
        print_status "Available commands:"
        print_status "• make run          - Start development server"
        print_status "• make test         - Run tests"
        print_status "• make lint         - Check code quality"
        print_status "• make help         - Show all available commands"
        print_status ""
        print_header "Documentation available in README.md"
        
        # Ask if user wants to start the server
        echo ""
        read -p "Do you want to start the server now? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Starting PDF Summarizer server..."
            python3 run.py --skip-models
        fi
    else
        print_error "Project validation failed. Please check the errors above."
        exit 1
    fi
}

# Error handling
set -e
trap 'print_error "Setup failed at line $LINENO"' ERR

# Run main function
main "$@"
