import os
from dotenv import load_dotenv

load_dotenv()

# ── Mistral API ──────────────────────────────────────────────
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_EMBED_MODEL = "mistral-embed"
MISTRAL_CHAT_MODEL = "mistral-small-latest"
MISTRAL_API_URL = "https://api.mistral.ai/v1"

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
SQLITE_DB_PATH = os.path.join(BASE_DIR, "chat.db")

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 50     # overlap between chunks

# ── RAG ──────────────────────────────────────────────────────
TOP_K_RESULTS = 5      # number of chunks to retrieve per query

# ── Ensure directories exist ─────────────────────────────────
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
