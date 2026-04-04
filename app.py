"""
RAG Chat — University Assistant
────────────────────────────────
Entry point. Starts the FastAPI server and serves the frontend.

Usage:
    python app.py
"""

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from config import MISTRAL_API_KEY
from backend.database import init_db
from backend.routes import router


# ── Startup ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    """Initialize database and check configuration on startup."""
    init_db()
    if not MISTRAL_API_KEY:
        print("\n⚠️  WARNING: MISTRAL_API_KEY is not set!")
        print("   Create a .env file with: MISTRAL_API_KEY=your_key_here")
        print("   Get your key at: https://console.mistral.ai/\n")
    else:
        print("\n✓ Mistral API key loaded")
    print("✓ Database initialized")
    print("✓ Server starting at http://localhost:8000\n")
    yield


# ── Initialize ───────────────────────────────────────────────

app = FastAPI(title="RAG Chat", lifespan=lifespan)

# Register API routes
app.include_router(router)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/")
def serve_frontend():
    """Serve the main chat interface."""
    return FileResponse("frontend/index.html")


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
