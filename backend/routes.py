"""
API Routes
──────────
FastAPI endpoints for the RAG chatbot:
  - Document upload & listing
  - Chat (RAG query)
  - Session management
"""

import os
import json
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from config import DOCUMENTS_DIR
from backend import database as db
from backend import document_processor
from backend import rag_engine
from backend import vector_store

router = APIRouter(prefix="/api")


# ── Request Models ───────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str

class SessionCreate(BaseModel):
    title: str = "New Chat"


# ── Document Endpoints ───────────────────────────────────────

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document, process it, embed chunks, and store in ChromaDB."""
    # Validate file type
    allowed_extensions = {".pdf", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Use PDF or TXT.")

    # Save file to documents folder
    file_path = os.path.join(DOCUMENTS_DIR, file.filename)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        # Extract text and split into chunks
        chunks = document_processor.process_document(file_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from this file.")

        # Embed and store in ChromaDB
        count = rag_engine.embed_and_store(chunks)

        # Record in SQLite
        db.add_document_record(file.filename, count)

        return {"filename": file.filename, "chunks": count, "message": f"Successfully processed {file.filename} into {count} chunks."}

    except Exception as e:
        # Clean up saved file on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
def list_documents():
    """List all uploaded documents with their chunk counts."""
    return db.get_document_records()


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a document and its chunks from the vector store."""
    vector_store.delete_by_source(filename)
    db.delete_document_record(filename)
    file_path = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": f"Deleted {filename}"}


# ── Chat Endpoints ───────────────────────────────────────────

@router.post("/chat")
def chat(request: ChatRequest):
    """Send a message and get a RAG-powered response."""
    # Verify session exists
    session = db.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    # Save user message
    db.add_message(request.session_id, "user", request.message)

    try:
        # Run RAG pipeline
        result = rag_engine.query(request.message)

        # Save assistant response with sources
        sources_json = json.dumps([
            {"source": s["source"], "text": s["text"][:200]}
            for s in result["sources"]
        ])
        db.add_message(request.session_id, "assistant", result["answer"], sources=sources_json)

        # Auto-title the session from the first message
        if session["title"] == "New Chat":
            title = request.message[:50] + ("..." if len(request.message) > 50 else "")
            db.update_session_title(request.session_id, title)

        return {
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Session Endpoints ────────────────────────────────────────

@router.get("/sessions")
def list_sessions():
    """List all chat sessions."""
    return db.get_sessions()


@router.post("/sessions")
def create_session(request: SessionCreate):
    """Create a new chat session."""
    return db.create_session(request.title)


@router.get("/sessions/{session_id}")
def get_session_messages(session_id: str):
    """Get all messages in a session."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    messages = db.get_messages(session_id)
    return {"session": session, "messages": messages}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a chat session and all its messages."""
    db.delete_session(session_id)
    return {"message": "Session deleted."}


# ── Status ───────────────────────────────────────────────────

@router.get("/status")
def status():
    """Quick health check with stats."""
    return {
        "status": "ok",
        "total_chunks": vector_store.get_chunk_count(),
        "total_documents": len(vector_store.get_all_sources()),
        "sources": vector_store.get_all_sources(),
    }
