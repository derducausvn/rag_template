"""
Database
────────
SQLite storage for chat sessions and messages.
Provides ordered, relational data alongside ChromaDB's vector store.
"""

import sqlite3
import uuid
from datetime import datetime
from config import SQLITE_DB_PATH


# ── Connection Helper ────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with row factory for dict-like access."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ── Schema Setup ─────────────────────────────────────────────

def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            title       TEXT NOT NULL,
            created_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          TEXT PRIMARY KEY,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
            content     TEXT NOT NULL,
            sources     TEXT,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS documents (
            id          TEXT PRIMARY KEY,
            filename    TEXT NOT NULL,
            chunk_count INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL
        );
    """)
    conn.commit()
    conn.close()


# ── Sessions ─────────────────────────────────────────────────

def create_session(title: str = "New Chat") -> dict:
    """Create a new chat session. Returns the session dict."""
    conn = get_connection()
    session = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.now().isoformat(),
    }
    conn.execute(
        "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
        (session["id"], session["title"], session["created_at"]),
    )
    conn.commit()
    conn.close()
    return session


def get_sessions() -> list[dict]:
    """Get all sessions, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    """Get a single session by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_session_title(session_id: str, title: str) -> None:
    """Update a session's title."""
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
    )
    conn.commit()
    conn.close()


def delete_session(session_id: str) -> None:
    """Delete a session and all its messages (cascade)."""
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()


# ── Messages ─────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str, sources: str = None) -> dict:
    """Add a message to a session. Returns the message dict."""
    conn = get_connection()
    message = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "role": role,
        "content": content,
        "sources": sources,
        "created_at": datetime.now().isoformat(),
    }
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (message["id"], message["session_id"], message["role"], message["content"], message["sources"], message["created_at"]),
    )
    conn.commit()
    conn.close()
    return message


def get_messages(session_id: str) -> list[dict]:
    """Get all messages in a session, in chronological order."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Documents ────────────────────────────────────────────────

def add_document_record(filename: str, chunk_count: int) -> dict:
    """Record a processed document. Returns the document dict."""
    conn = get_connection()
    doc = {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "chunk_count": chunk_count,
        "uploaded_at": datetime.now().isoformat(),
    }
    conn.execute(
        "INSERT INTO documents (id, filename, chunk_count, uploaded_at) VALUES (?, ?, ?, ?)",
        (doc["id"], doc["filename"], doc["chunk_count"], doc["uploaded_at"]),
    )
    conn.commit()
    conn.close()
    return doc


def get_document_records() -> list[dict]:
    """Get all document records."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM documents ORDER BY uploaded_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_document_record(filename: str) -> None:
    """Delete a document record by filename."""
    conn = get_connection()
    conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()
