# Apollyon RAG

A full-stack web application for document-based question answering using local LLMs with Retrieval-Augmented Generation (RAG) capabilities.

## Overview

Apollyon provides a local web interface for local LLMs allowing you to upload large documents, engage with the LLM in conversation, and get accurate answers based on the uploaded content using hybrid search (vector + keyword) and iterative RAG.

## Features

- **Document Upload & Processing** 📄: Upload text/markdown files (up to 10MB) which are automatically chunked and indexed
- **Hybrid Search** 🔗: Combines vector embeddings with TF-IDF keyword search for better retrieval
- **Iterative RAG** 🔄: Multiple retrieval iterations to gather comprehensive context before answering
- **Session Management** 💬: Multiple chat sessions with persistent conversation history - supports multiple users at the same time
- **Modern UI** 🎨: SvelteKit-based responsive frontend
- **FastAPI Backend** ⚡: Python backend with async streaming support

## Architecture

```
Frontend (SvelteKit) → Backend (FastAPI) → RAG System → Ollama LLM
                          ↓
                     Document Database
                    (HybridDB: Vector + Keyword)
```

## Prerequisites

1. **Ollama** 🦙: Install from [ollama.ai](https://ollama.ai/)
   ```bash
   # Install and start Ollama
   ollama serve
   ```
2. **Python 3.8+** 🐍 with pip
3. **Node.js 18+** 🟢 with npm

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Node.js dependencies
```bash
npm install
```

## Configuration

### Environment Setup

1. Ensure Ollama is running:
   ```bash
   ollama serve
   ```

2. The default configuration uses:
   - Model: `ministral-3:14b` (can be changed in `config.py`)
   - Embedding model: `all-MiniLM-L6-v2`
   - API endpoints:
     - Backend: `http://localhost:8000`
     - Frontend: `http://localhost:5173`
     - Ollama: `http://localhost:11434`

### Configuration Files

- **`main.py`**: FastAPI backend configuration
- **`vite.config.js`**: Frontend proxy configuration
- **`config.py`**: Model configuration 

## Usage

### 1. Start the backend
```bash
uvicorn main:app --reload --port 8000
```

### 2. Start the frontend

In a second terminal:

```bash
npm run dev
```

### 3. Open the application
Navigate to `http://localhost:5173` in your browser.

### 4. Upload documents
- Click the upload 🔗 button to add `.txt` or `.md` files
- Files are processed and indexed automatically
- Uploads may take a while depending on your hardware and file size ⏳

### 5. Ask questions
- Type questions in the chat interface
- The system will retrieve relevant context from uploaded documents
- Answers are generated using the local Ollama model 🤖

## Project Structure

```
├── frontend/               # SvelteKit application 
│   ├── src/                # Chat interface 
│   ├── static/             # Frontend assets 
│   └── package.json
├── backend/                # FastAPI application 
│   ├── main.py             # Main API server
│   ├── llm.py              # LLM wrapper classes 
│   ├── rag.py              # RAG pipeline 
│   ├── hdb.py              # Hybrid database 
│   ├── files.py            # File handling 
│   ├── stateful_llm.py     # Stateful LLM sessions
│   └── requirements.txt
├── example_data/            # Sample documents 
├── uploads/                 # User uploaded files 
└── README.md
```

## API Endpoints (backend server) 
- `POST /api/chat`: Stream chat completions 
- `POST /api/upload/`: Upload and process documents 

## Testing

You can run some tests using the example data:

```bash
# Test RAG system
python test_rag.py

# Test database
python test_db.py

# Test simplified RAG
python test_rag2.py
```

## Troubleshooting 🔧

1. **Ollama not running**:
   ```
   Error: Could not connect to Ollama. Is `ollama serve` running?
   ```
   Solution: Start Ollama with `ollama serve`

2. **File upload fails**:
   - Check file size (<10MB)
   - Ensure file extension is a supported format (code or text)
   - Verify write permissions in `uploads/` directory

3. **Slow response time**:
   - Ensure Ollama is warmed up
   - Use smaller model

## Acknowledgements

- [Ollama](https://ollama.ai/) for local LLM serving 
- [Sentence Transformers](https://www.sbert.net/) for embeddings 
- [LangChain](https://www.langchain.com/) for text splitting utilities
- [SvelteKit](https://kit.svelte.dev/) for frontend framework
- [FastAPI](https://fastapi.tiangolo.com/) for backend API
