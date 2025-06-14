# Study Buddy - PDF Q&A Assistant

A simple chatbot that helps you study by answering questions from your uploaded PDF notes. Upload your study materials and ask questions to better understand the content.

## Features

- Upload PDF documents containing your notes
- Ask questions about the uploaded content
- Get contextual answers based on your study materials
- Real-time chat interface
- WebSocket support for instant responses
- Chat history tracking

## Tech Stack

- Backend: FastAPI with Groq LLM
- Frontend: Streamlit
- Vector Store: FAISS
- Embeddings: HuggingFace Transformers
- Database: SQLite

## Setup

### Prerequisites

- Python 3.8+
- Pipenv
- Groq API key

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd study-buddy
```

2. Install dependencies
```bash
pipenv install
pipenv shell
```

3. Create a `.env` file and add your Groq API key
```
GROQ_API_KEY=your_groq_api_key_here
```

4. Run the backend server
```bash
python setup.py
```

5. Run the Streamlit frontend (in a new terminal)
```bash
streamlit run app.py
```

## Docker

Build and run with Docker:

```bash
docker build -t study-buddy .
docker run -p 8501:8501 -p 9999:9999 study-buddy
```

## Usage

1. Open the Streamlit app in your browser
2. Upload a PDF file containing your study notes
3. Start asking questions about the content
4. The chatbot will provide answers based on your uploaded materials

## API Endpoints

- `POST /upload-pdf/` - Upload PDF document
- `POST /chat/` - Send chat message
- `WebSocket /ws/` - Real-time chat connection

## License

MIT License