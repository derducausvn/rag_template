"""
Vector Store
─────────────
Wraps ChromaDB to store document chunks with their embeddings
and perform similarity search for RAG retrieval.
"""

import chromadb
from config import CHROMA_DB_DIR

# ── Initialize ChromaDB ──────────────────────────────────────

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(
    name="uni_docs",
    metadata={"hnsw:space": "cosine"},  # use cosine similarity
)


# ── Store Chunks ─────────────────────────────────────────────

def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Store document chunks with their embeddings in ChromaDB.

    Args:
        chunks: list of {"text": ..., "source": ..., "chunk_index": ...}
        embeddings: list of vectors from Mistral-embed (one per chunk)

    Returns:
        Number of chunks added.
    """
    if not chunks:
        return 0

    ids = [f"{c['source']}__chunk_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


# ── Search ───────────────────────────────────────────────────

def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the most similar chunks to a query embedding.

    Returns:
        List of {"text": ..., "source": ..., "distance": ...}
    """
    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    matches = []
    for i in range(len(results["documents"][0])):
        matches.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return matches


# ── Utilities ────────────────────────────────────────────────

def get_chunk_count() -> int:
    """Return total number of chunks stored."""
    return collection.count()


def get_all_sources() -> list[str]:
    """Return a list of unique source filenames in the store."""
    if collection.count() == 0:
        return []
    all_data = collection.get(include=["metadatas"])
    sources = set(m["source"] for m in all_data["metadatas"])
    return sorted(sources)


def delete_by_source(source: str) -> None:
    """Delete all chunks from a specific source document."""
    if collection.count() == 0:
        return
    all_data = collection.get(include=["metadatas"])
    ids_to_delete = [
        all_data["ids"][i]
        for i, m in enumerate(all_data["metadatas"])
        if m["source"] == source
    ]
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
