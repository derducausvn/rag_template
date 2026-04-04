"""
Document Processor
──────────────────
Reads PDF and TXT files, extracts text, and splits into chunks
for embedding. Each chunk includes metadata (source file, chunk index).
"""

import os
from PyPDF2 import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP


# ── Text Extraction ──────────────────────────────────────────

def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def extract_text_from_txt(file_path: str) -> str:
    """Read a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(file_path: str) -> str:
    """Extract text from a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Chunking ─────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks.

    Example with chunk_size=10, overlap=3:
      "Hello world, how are you today"
      → ["Hello worl", "orld, how ", "how are yo", "re you tod", "ou today"]
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── Main Processing Function ─────────────────────────────────

def process_document(file_path: str) -> list[dict]:
    """
    Process a single document: extract text → split into chunks.

    Returns a list of dicts:
      [{"text": "chunk text...", "source": "handbook.pdf", "chunk_index": 0}, ...]
    """
    text = extract_text(file_path)
    if not text.strip():
        return []

    chunks = split_into_chunks(text)
    filename = os.path.basename(file_path)

    return [
        {
            "text": chunk,
            "source": filename,
            "chunk_index": i,
        }
        for i, chunk in enumerate(chunks)
    ]
