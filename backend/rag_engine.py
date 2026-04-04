"""
RAG Engine
──────────
Core logic that ties together:
  1. Mistral-embed  → create vectors
  2. ChromaDB       → store & search vectors
  3. Mistral-small  → generate answers from retrieved context
"""

import httpx
from config import MISTRAL_API_KEY, MISTRAL_EMBED_MODEL, MISTRAL_CHAT_MODEL, MISTRAL_API_URL, TOP_K_RESULTS
from backend import vector_store


# ── Mistral API Calls ────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Call Mistral-embed to convert texts into vectors.

    Args:
        texts: list of strings to embed

    Returns:
        list of vectors (one per input text)
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set. Check your .env file.")

    response = httpx.post(
        f"{MISTRAL_API_URL}/embeddings",
        headers=_headers(),
        json={"model": MISTRAL_EMBED_MODEL, "input": texts},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]


def chat_completion(system_prompt: str, user_message: str) -> str:
    """
    Call Mistral-small-latest to generate a response.

    Args:
        system_prompt: instructions + retrieved context
        user_message: the user's question

    Returns:
        The assistant's reply as a string.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set. Check your .env file.")

    response = httpx.post(
        f"{MISTRAL_API_URL}/chat/completions",
        headers=_headers(),
        json={
            "model": MISTRAL_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


# ── Embed & Store Documents ──────────────────────────────────

def embed_and_store(chunks: list[dict]) -> int:
    """
    Embed document chunks with Mistral and store in ChromaDB.

    Args:
        chunks: list of {"text": ..., "source": ..., "chunk_index": ...}

    Returns:
        Number of chunks stored.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]

    # Embed in batches of 25 (Mistral API limit)
    all_embeddings = []
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)

    return vector_store.add_chunks(chunks, all_embeddings)


# ── RAG Query ────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a helpful university assistant. Answer the student's question based ONLY on the provided context.

Rules:
- Use only the information from the context below to answer.
- If the context doesn't contain enough information, say "I don't have enough information in my documents to answer this question."
- Cite the source document when possible (e.g. "According to handbook.pdf...").
- Be concise and direct.

Context:
{context}"""


def query(question: str) -> dict:
    """
    Full RAG pipeline:
      1. Embed the question
      2. Search ChromaDB for similar chunks
      3. Build a prompt with retrieved context
      4. Get Mistral's answer

    Returns:
        {"answer": "...", "sources": [{"text": ..., "source": ..., "distance": ...}]}
    """
    # Step 1: Embed the question
    question_embedding = get_embeddings([question])[0]

    # Step 2: Search for relevant chunks
    matches = vector_store.search(question_embedding, top_k=TOP_K_RESULTS)

    if not matches:
        return {
            "answer": "No documents have been uploaded yet. Please upload some documents first.",
            "sources": [],
        }

    # Step 3: Build context from matched chunks
    context_parts = []
    for i, match in enumerate(matches, 1):
        context_parts.append(f"[{i}] (Source: {match['source']})\n{match['text']}")
    context = "\n\n".join(context_parts)

    system_prompt = SYSTEM_TEMPLATE.format(context=context)

    # Step 4: Generate answer
    answer = chat_completion(system_prompt, question)

    return {
        "answer": answer,
        "sources": matches,
    }
