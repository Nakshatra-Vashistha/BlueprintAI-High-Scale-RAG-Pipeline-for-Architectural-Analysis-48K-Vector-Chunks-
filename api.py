from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_pipeline import RAGPipeline
from src.config import config

app = FastAPI(title="Architecture RAG API", description="API for Architecture RAG with Cloud Auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG pipeline
rag = RAGPipeline()

# ==========================================
# Pydantic Models (Data Validation)
# ==========================================
class QueryRequest(BaseModel):
    query: str
    n_results: int = 5  # Added to match frontend payload

class QueryResponse(BaseModel):
    answer: str
    sources: list

class UserCredentials(BaseModel):
    email: str
    password: str

# New schemas for the Interactive Session feature
class ConversationItem(BaseModel):
    query: str
    answer: str

class InteractiveRequest(BaseModel):
    query: str
    conversation_history: List[ConversationItem] = []
    n_results: int = 5

# ==========================================
# Supabase Authentication Endpoints
# ==========================================
@app.post("/signup")
async def signup(credentials: UserCredentials):
    if not rag.db.supabase:
        raise HTTPException(status_code=500, detail="Supabase client is not configured.")
    try:
        response = rag.db.supabase.auth.sign_up({
            "email": credentials.email,
            "password": credentials.password
        })
        return {"message": "User created successfully", "user": response.user}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Signup failed: {str(e)}")

@app.post("/login")
async def login(credentials: UserCredentials):
    if not rag.db.supabase:
        raise HTTPException(status_code=500, detail="Supabase client is not configured.")
    try:
        response = rag.db.supabase.auth.sign_in_with_password({
            "email": credentials.email,
            "password": credentials.password
        })
        return {"access_token": response.session.access_token, "user": response.user}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid credentials: {str(e)}")

# ==========================================
# Core RAG Endpoints
# ==========================================
@app.post("/query", response_model=QueryResponse)
async def query_research_papers(request: QueryRequest):
    try:
        result = rag.query(request.query, n_results=request.n_results)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# FIXED: Fully contextualized interactive endpoint
@app.post("/api/interactive")
async def interactive_query(request: InteractiveRequest):
    try:
        # 1. Build a contextualized query string so the AI remembers the chat
        if request.conversation_history:
            # Grab the last 3 exchanges so we don't overload the token limit
            recent_history = request.conversation_history[-3:]
            
            history_text = "\n".join(
                [f"User: {item.query}\nAI: {item.answer}" for item in recent_history]
            )
            
            # Prepend the history to the new query
            search_query = f"Previous Conversation:\n{history_text}\n\nNew Question: {request.query}\n\nAnswer the New Question based on the context of the Previous Conversation."
        else:
            search_query = request.query

        # 2. Pass the compiled contextual query to your RAG pipeline
        result = rag.query(search_query, n_results=request.n_results)
        
        # 3. Append the new exchange to the history array for the frontend
        new_history = request.conversation_history + [
            ConversationItem(query=request.query, answer=result["answer"])
        ]
        
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),  # .get() prevents KeyError crashes!
            "context": result.get("context", []),  # .get() prevents KeyError crashes!
            "query": request.query,
            "conversation_history": new_history
        }
    except Exception as e:
        print(f"Server Error in Interactive Query: {str(e)}") # Prints to your terminal for debugging
        raise HTTPException(status_code=500, detail=f"Interactive query failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Cloud Architecture RAG API is running"}

@app.get("/stats")
async def get_stats():
    try:
        stats = rag.db.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/init")
async def initialize_database():
    try:
        rag.initialize_database(config.JSONL_FILES)
        return {"message": "Pinecone Vector Database initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)