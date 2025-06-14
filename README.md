# Study Buddy

A PDF-based question-answering system that helps students study by allowing them to upload their notes and ask questions interactively. Built with FastAPI backend and Streamlit frontend.

## Screenshots

### Main Interface
![Upload Interface](https://drive.google.com/file/d/1H28GLOVI7XJl-7bXJRUOv-gYwnAJ46Rc/view?usp=sharing)

### Chat Interface
![Chat Interface](https://drive.google.com/file/d/1pKQwUqe-izLfvm9tIxBCBKSrk_SicMiY/view?usp=sharing)

## Features

- Upload PDF documents containing study notes
- Interactive chat interface for asking questions
- Vector-based document retrieval using FAISS
- Conversation history tracking
- Rate limiting and WebSocket support
- SQLite database for persistence

## Architecture

![System Architecture](images/architecture-diagram.png)

The system consists of two main components:

### Backend (setup.py)
- **FastAPI** server running on port 9999
- **Groq LLM** (llama-3.3-70b-versatile) for answer generation
- **HuggingFace Embeddings** (sentence-transformers/all-MiniLM-L6-v2) for document vectorization
- **FAISS** vector store for similarity search
- **SQLite** database for storing PDF content and chat history
- **WebSocket** support for real-time chat

### Frontend (app.py)
- **Streamlit** web interface
- PDF upload functionality
- Chat interface with message history
- Session state management

## Setup Instructions

### Prerequisites
- Python 3.8+
- Pipenv
- Groq API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd study-buddy
```

2. Install dependencies using pipenv:
```bash
pipenv install
pipenv shell
```

3. Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the backend server:
```bash
python setup.py
```

5. In a new terminal, run the frontend:
```bash
streamlit run app.py
```

### Docker Setup

If you have Docker configured for this project:

```bash
docker build -t study-buddy .
docker run -p 8501:8501 -p 9999:9999 study-buddy
```

## Usage Guide

### 1. Upload PDF Notes
![PDF Upload Process](images/upload-process.png)

- Click "Browse files" to select your PDF study notes
- Click "Upload PDF" to process the document
- Wait for confirmation message showing text length

### 2. Ask Questions
![Chat Example](images/chat-example.png)

- Type your question in the chat input field
- The system will search through your uploaded notes
- Receive detailed answers based on the document content

### 3. Chat History
- All conversations are saved and displayed in the chat interface
- Session state maintains conversation context

## API Reference

### Upload PDF
```http
POST /upload-pdf/
Content-Type: multipart/form-data

Response:
{
  "message": "PDF uploaded and processed successfully",
  "text_length": 15420,
  "chunks_created": 45
}
```

### Chat
```http
POST /chat/
Content-Type: application/json

{
  "message": "What is the main topic of chapter 1?"
}

Response:
{
  "response": "Based on the uploaded document...",
  "sources": ["Document excerpt 1...", "Document excerpt 2..."]
}
```

### WebSocket Chat
```
WS /ws/

Send: "Your question here"
Receive: {
  "answer": "Response from the system",
  "sources": ["source1", "source2"],
  "session_id": "session_123"
}
```

## Configuration

Key configuration options in `Config` class:

- `GROQ_MODEL`: "llama-3.3-70b-versatile"
- `EMBEDDING_MODEL`: "sentence-transformers/all-MiniLM-L6-v2"
- `CHUNK_SIZE`: 500 characters
- `CHUNK_OVERLAP`: 50 characters
- `MAX_CONVERSATION_HISTORY`: 5 messages

## Rate Limits

- PDF Upload: 5 requests per minute
- Chat: 20 requests per minute

## File Structure

```
study-buddy/
├── app.py              # Streamlit frontend
├── setup.py            # FastAPI backend
├── .env                # Environment variables
├── pdf_qa.db           # SQLite database (auto-created)
├── vectorstore/        # FAISS vector store (auto-created)
└── README.md
```

## Database Schema

### pdf_content
- `id`: Primary key
- `filename`: Original PDF filename
- `upload_date`: Upload timestamp
- `content`: Extracted text content
- `chunk_count`: Number of text chunks created

### chat_history
- `id`: Primary key
- `created_at`: Timestamp
- `question`: User question
- `answer`: System response
- `sources`: JSON array of source excerpts

## Assumptions

1. **PDF Format**: Only PDF files are supported for upload
2. **Text-based PDFs**: The system works best with text-based PDFs (not scanned images)
3. **English Language**: Optimized for English language content
4. **Single Document**: Currently supports one PDF at a time (new uploads replace previous ones)
5. **Local Storage**: All data is stored locally in SQLite database
6. **Internet Connection**: Required for Groq API and HuggingFace model downloads
7. **Resource Requirements**: Moderate CPU/RAM usage for embedding generation

## Troubleshooting

- **"No PDF content found"**: Upload a PDF file before asking questions
- **Upload failed**: Ensure the file is a valid PDF with extractable text
- **API errors**: Verify your Groq API key is correctly set in the `.env` file
- **Port conflicts**: Ensure ports 8501 and 9999 are available

## Dependencies

Key libraries used:
- FastAPI
- Streamlit
- LangChain
- FAISS
- HuggingFace Transformers
- Groq
- PyPDF
- SQLite3