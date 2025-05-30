# Cloby - AI School Assistant

Cloby is an AI-powered school assistant that helps students and teachers with academic support, data analysis, and learning tools.

## Prerequisites

1. Python 3.8 or higher
2. Ollama installed and running locally

## Setup Instructions

1. Install Ollama:
   - Visit https://ollama.ai/download
   - Download and install Ollama for your operating system
   - Run Ollama and pull the Mistral model:
     ```bash
     ollama pull mistral
     ```

2. Create a Python virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Cloby

1. Make sure Ollama is running in the background
2. Run the application:
   ```bash
   python cloby.py
   ```
3. Open the provided URL in your web browser (usually http://localhost:7860)

## Features

- Subject support (math, science, literature, etc.)
- Homework problem solving with steps
- Quiz and flashcard generation
- Study plan creation
- Citation and research help
- Student data analysis

## Note

 