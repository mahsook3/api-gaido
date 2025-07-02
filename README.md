# RAG Question Answering System

This project provides a Retrieval-Augmented Generation (RAG) system that allows you to ask questions about PDF documents and get AI-powered answers.

## Files

- `app.py` - Core RAG system that processes PDFs and creates the vector database
- `gradio_ui.py` - Web-based user interface using Gradio
- `requirements.txt` - Python dependencies
- `chroma_db/` - Vector database storage (created after running app.py)

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file with your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Initialize the database:**
   Run the core application first to download PDFs and create the vector database:
   ```bash
   python app.py
   ```

4. **Launch the web interface:**
   ```bash
   python gradio_ui.py
   ```

## Usage

### Command Line Interface (app.py)
- Processes PDF documents and creates embeddings
- Provides an interactive command-line interface for asking questions

### Web Interface (gradio_ui.py)
- Beautiful web-based interface accessible via browser
- Adjustable number of relevant passages to retrieve
- Shows both the AI-generated answer and source passages
- Example questions provided for easy testing

## Features

- **Document Processing**: Automatically downloads and processes PDF documents
- **Vector Search**: Uses ChromaDB for efficient similarity search
- **AI Generation**: Powered by Google's Gemini AI for natural language responses
- **Web Interface**: User-friendly Gradio interface with real-time responses
- **Source Attribution**: Shows relevant passages used to generate answers

## Accessing the Web Interface

After running `python gradio_ui.py`, the interface will be available at:
- Local: http://localhost:7860
- The terminal will show the exact URL to access the interface

## Tips for Better Results

- Be specific in your questions
- Try different phrasings if you don't get good results
- Use keywords related to your documents
- Adjust the number of relevant passages (1-5) based on your needs
