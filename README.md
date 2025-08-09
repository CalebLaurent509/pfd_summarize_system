# PDF Summarize - Multilingual Document Summarization System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.0-green.svg)](https://flask.palletsprojects.com/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.47.1-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

An intelligent document summarization system that automatically detects text language, translates if necessary, generates high-quality summaries, and returns results in the original language. Supports both text input and PDF file processing.

## Features

- **PDF Processing**: Extract and process text from PDF documents
- **Automatic Language Detection**: Identifies text language automatically
- **Multilingual Translation**: Support for Romance languages (French, Spanish, Italian, Portuguese, Romanian, Catalan) to English and vice versa
- **Intelligent Summarization**: Uses Facebook's BART model for high-quality summary generation
- **REST API**: Simple interface via Flask API
- **Complete Pipeline**: End-to-end automated text processing
- **Text Preprocessing**: Advanced text cleaning and normalization
- **CORS Support**: Compatible with web frontend applications

## Technologies Used

- **Python 3.8+**
- **Flask**: Web framework for REST API
- **Transformers (Hugging Face)**: Natural language processing models
- **PyTorch**: Machine learning backend
- **NLTK**: Natural language toolkit for preprocessing
- **PyPDF2**: PDF text extraction
- **langdetect**: Automatic language detection
- **BART**: Facebook's summarization model
- **MarianMT**: Helsinki-NLP translation models

## Project Structure

```text
pdf_summarize/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── .gitignore               # Git ignore rules
├── core/                    # Core functionality
│   ├── __init__.py
│   ├── generator.py         # Main summary generation
│   ├── logic.py            # Processing pipeline logic
│   └── summarizer.py       # BART summarization model
├── src/                     # Source modules
│   ├── __init__.py
│   └── translator_models.py # Translation models handler
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── languages_detect.py # Language detection
│   ├── pdf_extractor.py    # PDF text extraction
│   └── preprocessing.py    # Text preprocessing
└── models/                  # Model storage directory (auto-created)
    ├── Bart/               # BART model cache
    ├── translation_models/ # Translation model cache
    └── nltk_data/         # NLTK data
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

### Local Installation

1. **Clone the repository**

```bash
git clone https://github.com/CalebLaurent509/pdf_summarize_system.git
cd pdf_summarize_system
```

2. **Create virtual environment**

```bash
python -m venv env
source env/bin/activate  # On Linux/Mac
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### POST `/summarize`

Generate a summary for the provided text.

**Request Body:**

```json
{
    "text": "Your text to summarize here..."
}
```

**Response:**

```json
{
    "lang": "en",
    "text": "Generated summary in the original language"
}
```

### Command Line Usage

```bash
# Process a PDF file directly
python core/generator.py
```

Make sure to place your PDF file as `sample.pdf` in the `core/` directory.

### API Usage Examples

#### Using curl

```bash
curl -X POST http://localhost:5000/summarize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your long text to summarize goes here..."
  }'
```

#### Using Python requests

```python
import requests

url = "http://localhost:5000/summarize"
data = {
    "text": "Your long text to summarize here..."
}

response = requests.post(url, json=data)
result = response.json()
print(f"Detected language: {result['lang']}")
print(f"Summary: {result['text']}")
```

#### Using JavaScript fetch

```javascript
const response = await fetch('http://localhost:5000/summarize', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        text: 'Your long text to summarize here...'
    })
});

const result = await response.json();
console.log('Detected language:', result.lang);
console.log('Summary:', result.text);
```

## Supported Languages

The system supports the following languages:

- **English** (en) - Direct processing
- **French** (fr)
- **Spanish** (es)
- **Italian** (it)
- **Portuguese** (pt)
- **Romanian** (ro)
- **Catalan** (ca)

## Configuration

### Model Parameters

You can adjust summarization parameters in `core/generator.py`:

- `input_max_length`: Maximum input text length (default: 1024)
- `sum_max_length`: Maximum summary length (default: 200)
- `sum_min_length`: Minimum summary length (default: 20)
- `num_beams`: Number of beams for generation (default: 2)

### Models Used

- **Summarization**: `facebook/bart-large-cnn`
- **Translation Romance → English**: `Helsinki-NLP/opus-mt-ROMANCE-en`
- **Translation English → Romance**: `Helsinki-NLP/opus-mt-en-ROMANCE`

## Processing Pipeline

1. **PDF Text Extraction** (if applicable): Extract text from PDF documents
2. **Text Preprocessing**: Clean and normalize text data
3. **Language Detection**: Automatically identify source language
4. **Language Validation**: Check if language is supported
5. **Translation** (if needed): Convert to English for processing
6. **Summarization**: Generate summary using BART model
7. **Back Translation** (if needed): Convert summary to original language
8. **Response**: Return summary in original language

## Deployment

### Heroku Deployment

The project is ready for Heroku deployment:

1. Create a Heroku application
2. Connect your repository
3. Deploy automatically

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

Build and run:

```bash
docker build -t pdf-summarize .
docker run -p 5000:5000 pdf-summarize
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed

```bash
pip install -r requirements.txt
```

2. **Memory Issues**: BART models are large. Ensure at least 4GB RAM available.

3. **NLTK Data Missing**: Download required NLTK data

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
```

4. **PDF Processing Errors**: Ensure PDF files are text-based, not image-based.

5. **Language Not Supported**: The system will return an error if the detected language is not in the supported list.

### Performance Optimization

- **Model Caching**: Models are automatically cached locally after first download
- **Text Chunking**: Long texts are automatically truncated to fit model limits
- **Preprocessing**: Text is cleaned and normalized for better results

## Performance Metrics

- **Processing Time**: 5-15 seconds depending on text length
- **Memory Usage**: ~2-4GB for model loading
- **Supported Languages**: 7 languages
- **Summary Quality**: High-quality summaries using state-of-the-art BART model
- **File Support**: PDF text extraction and plain text processing

## Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Code formatting
black .
flake8 .
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Facebook AI](https://ai.facebook.com/) for the BART model
- [Helsinki-NLP](https://github.com/Helsinki-NLP) for translation models
- [NLTK](https://www.nltk.org/) for natural language processing tools
- The Python community for excellent libraries

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: Models will be automatically downloaded on first run. Ensure you have a stable internet connection and sufficient disk space (approximately 1.5GB for all models).

If this project helped you, please give it a star!
