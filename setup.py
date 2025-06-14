import os
import logging
from typing import List, Dict, Optional
from contextlib import contextmanager
import sqlite3
import json
import asyncio
from datetime import datetime
import pypdf
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, Request
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

class Config:
    """Configuration class for the application"""
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    GROQ_MODEL = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    DB_PATH = 'pdf_qa.db'
    VECTOR_STORE_PATH = "vectorstore/db_faiss"
    MAX_CONTEXT_LENGTH = 4000
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    MAX_CONVERSATION_HISTORY = 5


app = FastAPI(title="PDF Q&A Assistant", version="1.0.0")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Pydantic models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

class UploadResponse(BaseModel):
    message: str
    text_length: int
    chunks_created: int

# Database context manager
@contextmanager
def get_db():
    """Database connection context manager"""
    conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize database tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # PDF content table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP NOT NULL,
            content TEXT NOT NULL,
            chunk_count INTEGER DEFAULT 0
        )''')
        
        # Chat history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT
        )''')
        
        conn.commit()

class LLMService:
    """Service class for LLM operations"""
    
    def __init__(self):
        self.llm = None
        self.qa_chain = None
        self.embeddings = None
        self.vectorstore = None
        self._initialize_llm()
        self._initialize_embeddings()
    
    def _initialize_llm(self):
        """Initialize Groq LLM"""
        try:
            self.llm = ChatGroq(
                groq_api_key=Config.GROQ_API_KEY,
                model_name=Config.GROQ_MODEL,
                temperature=0.3,
                max_tokens=1024
            )
            logger.info("Groq LLM initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def create_vectorstore(self, documents: List[Document]) -> int:
        """Create or update vector store from documents"""
        try:
            # Create vector store
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save vector store
            os.makedirs(os.path.dirname(Config.VECTOR_STORE_PATH), exist_ok=True)
            self.vectorstore.save_local(Config.VECTOR_STORE_PATH)
            
            logger.info(f"Vector store created with {len(documents)} documents")
            return len(documents)
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def load_vectorstore(self):
        """Load existing vector store"""
        try:
            if os.path.exists(Config.VECTOR_STORE_PATH):
                self.vectorstore = FAISS.load_local(
                    Config.VECTOR_STORE_PATH, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False
    
    def setup_qa_chain(self):
        """Setup QA chain with custom prompt"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        custom_prompt = PromptTemplate(
            template="""Use the following context to answer the user's question thoroughly and accurately.
            If the answer is present in the context, provide a detailed and well-structured response.
            If you don't know the answer, clearly state that you don't know instead of making up information.
            Do not include information beyond what is given in the context.
            
            Context: {context}
            Question: {question}
            
            Provide a clear and informative response:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={'k': 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={'prompt': custom_prompt}
        )
        
        logger.info("QA chain setup completed")
    
    def get_answer(self, question: str) -> Dict:
        """Get answer from QA chain"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized")
        
        try:
            response = self.qa_chain.invoke({'query': question})
            sources = [doc.page_content[:200] + "..." for doc in response.get("source_documents", [])]
            
            return {
                "answer": response["result"],
                "sources": sources
            }
        except Exception as e:
            logger.error(f"Error getting answer: {e}")
            raise

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_contexts: Dict[str, List[Dict]] = {}
    
    async def connect(self, websocket: WebSocket) -> str:
        await websocket.accept()
        session_id = f"session_{len(self.active_connections) + 1}_{datetime.now().timestamp()}"
        self.active_connections[session_id] = websocket
        self.session_contexts[session_id] = []
        return session_id
    
    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        self.session_contexts.pop(session_id, None)
    
    def get_session_context(self, session_id: str) -> List[Dict]:
        return self.session_contexts.get(session_id, [])
    
    def update_session_context(self, session_id: str, question: str, answer: str):
        if session_id in self.session_contexts:
            self.session_contexts[session_id].append({
                "question": question,
                "answer": answer
            })
            # Keep only last N exchanges
            self.session_contexts[session_id] = self.session_contexts[session_id][-Config.MAX_CONVERSATION_HISTORY:]

# Initialize services
llm_service = LLMService()
manager = ConnectionManager()

def extract_text_from_pdf(file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = pypdf.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")

def create_document_chunks(text: str, filename: str) -> List[Document]:
    """Create document chunks from text"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    documents = [
        Document(
            page_content=chunk,
            metadata={"source": filename, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]
    
    return documents

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize database and load existing vector store"""
    init_db()
    llm_service.load_vectorstore()
    if llm_service.vectorstore:
        llm_service.setup_qa_chain()

@app.post("/upload-pdf/", response_model=UploadResponse)
@limiter.limit("5/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    """Upload and process PDF file"""
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Extract text from PDF
        with open(temp_path, "rb") as pdf_file:
            extracted_text = extract_text_from_pdf(pdf_file)
        
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # Create document chunks
        documents = create_document_chunks(extracted_text, file.filename)
        
        # Create vector store
        chunk_count = llm_service.create_vectorstore(documents)
        
        # Setup QA chain
        llm_service.setup_qa_chain()
        
        # Store in database
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO pdf_content (filename, upload_date, content, chunk_count) VALUES (?, ?, ?, ?)",
                (file.filename, datetime.now().isoformat(), extracted_text, chunk_count)
            )
            conn.commit()
        
        return UploadResponse(
            message="PDF uploaded and processed successfully",
            text_length=len(extracted_text),
            chunks_created=chunk_count
        )
    
    except Exception as e:
        logger.error(f"Error processing PDF upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat/", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, chat_request: ChatRequest):
    """Chat endpoint for Q&A"""
    if not llm_service.qa_chain:
        raise HTTPException(status_code=404, detail="No PDF content found. Please upload a PDF first.")
    
    try:
        # Get answer from QA chain
        result = llm_service.get_answer(chat_request.message)
        
        # Store chat history
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (created_at, question, answer, sources) VALUES (?, ?, ?, ?)",
                (
                    datetime.now().isoformat(),
                    chat_request.message,
                    result["answer"],
                    json.dumps(result["sources"])
                )
            )
            conn.commit()
        
        return ChatResponse(
            response=result["answer"],
            sources=result["sources"]
        )
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    if not llm_service.qa_chain:
        await websocket.close(code=4004, reason="No PDF content found")
        return
    
    session_id = await manager.connect(websocket)
    logger.info(f"WebSocket connected: {session_id}")
    
    try:
        while True:
            try:
                # Receive message with timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=300.0)
                
                # Get answer from QA chain
                result = llm_service.get_answer(message)
                answer = result["answer"]
                
                # Update session context
                manager.update_session_context(session_id, message, answer)
                
                # Store in database
                with get_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO chat_history (created_at, question, answer, sources) VALUES (?, ?, ?, ?)",
                        (
                            datetime.now().isoformat(),
                            message,
                            answer,
                            json.dumps(result["sources"])
                        )
                    )
                    conn.commit()
                
                # Send response
                await websocket.send_text(json.dumps({
                    "answer": answer,
                    "sources": result["sources"],
                    "session_id": session_id
                }))
                
            except asyncio.TimeoutError:
                await websocket.close(code=4000, reason="Session timeout")
                break
            except Exception as e:
                logger.error(f"Error in websocket message handling: {e}")
                await websocket.send_text(json.dumps({"error": str(e)}))
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(session_id)
        logger.info(f"WebSocket disconnected: {session_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)