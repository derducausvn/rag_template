"""
RAG Benchmark Runner
────────────────────
Automated benchmarking for the RAG chatbot across different configurations.
Measures latency, token usage, and estimated cost per query.
Generates a survey HTML file for human quality evaluation.

Usage:
    python benchmark.py
"""

import os
import time
import json
import csv
import httpx
import chromadb
from datetime import datetime

from config import MISTRAL_API_KEY, MISTRAL_EMBED_MODEL, MISTRAL_API_URL, DOCUMENTS_DIR
from backend.document_processor import extract_text, split_into_chunks


# ═══════════════════════════════════════════════════════════════
#  EDIT THIS SECTION — your questions and configurations
# ═══════════════════════════════════════════════════════════════

BENCHMARK_QUESTIONS = [
    "What are the admission requirements for the Computer Science program?",
    "How does the grading system work at the university?",
    "What scholarships are available for international students?",
    "What is the process for registering courses each semester?",
    "Who should I contact for academic advising in the engineering faculty?",
]

BENCHMARK_CONFIGS = [
    # ── Group A: Vary chunk size (isolate chunking granularity) ──
    #    Small chunks = more precise retrieval, less context per chunk
    #    Large chunks = more context per chunk, but noisier retrieval
    {
        "name": "A1-small-chunk",
        "top_k": 5,
        "chunk_size": 256,
        "chunk_overlap": 25,
        "chat_model": "mistral-small-latest",
    },
    {
        "name": "A2-medium-chunk",   # ← baseline (your current default)
        "top_k": 5,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chat_model": "mistral-small-latest",
    },
    {
        "name": "A3-large-chunk",
        "top_k": 5,
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "chat_model": "mistral-small-latest",
    },

    # ── Group B: Vary top_k (isolate retrieval depth) ────────────
    #    Fewer chunks = faster, cheaper, but may miss info
    #    More chunks = richer context, but more noise + tokens
    {
        "name": "B1-topk3",
        "top_k": 3,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chat_model": "mistral-small-latest",
    },
    # B-baseline = A2-medium-chunk (same params, no need to re-run)
    {
        "name": "B2-topk10",
        "top_k": 10,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chat_model": "mistral-small-latest",
    },

    # ── Group C: Vary model (isolate model capability) ───────────
    #    Both are free-tier models with different architectures
    # C-baseline = A2-medium-chunk (mistral-small, same params)
    {
        "name": "C1-nemo",
        "top_k": 5,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chat_model": "open-mistral-nemo",
    },

    # ── Group D: Combined "best" config ──────────────────────────
    {
        "name": "D1-max-all",
        "top_k": 10,
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "chat_model": "open-mistral-nemo",
    },
]

# Mistral pricing per 1M tokens — update if prices change
# https://docs.mistral.ai/getting-started/pricing/
PRICING_PER_1M_TOKENS = {
    "mistral-small-latest":  {"input": 0.1,  "output": 0.3},
    "open-mistral-nemo":     {"input": 0.15, "output": 0.15},
    "mistral-embed":         {"input": 0.1,  "output": 0.0},
}

# Output directory for results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results")


# ═══════════════════════════════════════════════════════════════
#  Mistral API helpers (with token tracking)
# ═══════════════════════════════════════════════════════════════

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
    }


def get_embeddings(texts: list[str]) -> tuple[list[list[float]], dict]:
    """Embed texts and return (embeddings, usage_dict)."""
    response = httpx.post(
        f"{MISTRAL_API_URL}/embeddings",
        headers=_headers(),
        json={"model": MISTRAL_EMBED_MODEL, "input": texts},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = [item["embedding"] for item in data["data"]]
    usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return embeddings, usage


def chat_completion(system_prompt: str, user_message: str, model: str) -> tuple[str, dict]:
    """Chat completion and return (answer, usage_dict)."""
    response = httpx.post(
        f"{MISTRAL_API_URL}/chat/completions",
        headers=_headers(),
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    return answer, usage


# ═══════════════════════════════════════════════════════════════
#  Document processing & vector store per config
# ═══════════════════════════════════════════════════════════════

def get_all_document_paths() -> list[str]:
    """Return paths of all PDF/TXT files in the documents folder."""
    paths = []
    for fname in os.listdir(DOCUMENTS_DIR):
        ext = os.path.splitext(fname)[1].lower()
        if ext in (".pdf", ".txt"):
            paths.append(os.path.join(DOCUMENTS_DIR, fname))
    return sorted(paths)


def build_collection(chunk_size: int, chunk_overlap: int, chroma_client: chromadb.Client) -> str:
    """
    Chunk all documents with the given parameters and store in a
    dedicated ChromaDB collection.  Returns the collection name.
    """
    collection_name = f"bench_cs{chunk_size}_co{chunk_overlap}"

    # Reuse if already built in this run
    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        return collection_name

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    doc_paths = get_all_document_paths()
    if not doc_paths:
        print("  ⚠  No documents found in", DOCUMENTS_DIR)
        return collection_name

    all_chunks = []
    for path in doc_paths:
        text = extract_text(path)
        if not text.strip():
            continue
        pieces = split_into_chunks(text, chunk_size=chunk_size, overlap=chunk_overlap)
        filename = os.path.basename(path)
        for i, piece in enumerate(pieces):
            all_chunks.append({"text": piece, "source": filename, "chunk_index": i})

    print(f"  → {len(all_chunks)} chunks from {len(doc_paths)} documents "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap})")

    # Embed in batches
    texts = [c["text"] for c in all_chunks]
    all_embeddings = []
    batch_size = 25
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs, _ = get_embeddings(batch)
        all_embeddings.extend(embs)

    ids = [f"{c['source']}__chunk_{c['chunk_index']}" for c in all_chunks]
    documents = texts
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in all_chunks]

    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection_name


# ═══════════════════════════════════════════════════════════════
#  RAG query with full metrics
# ═══════════════════════════════════════════════════════════════

SYSTEM_TEMPLATE = """You are a helpful university assistant. Answer the student's question based ONLY on the provided context.

Rules:
- Use only the information from the context below to answer.
- If the context doesn't contain enough information, say "I don't have enough information in my documents to answer this question."
- Cite the source document when possible (e.g. "According to handbook.pdf...").
- Be concise and direct.

Context:
{context}"""


def run_query(question: str, collection, top_k: int, chat_model: str) -> dict:
    """
    Execute a single RAG query and return detailed metrics.
    """
    metrics = {}

    # Step 1: Embed the question
    t0 = time.perf_counter()
    q_embedding, embed_usage = get_embeddings([question])
    embed_time = time.perf_counter() - t0
    q_embedding = q_embedding[0]

    # Step 2: Retrieve relevant chunks
    t1 = time.perf_counter()
    count = collection.count()
    if count == 0:
        return {"error": "Collection is empty"}
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, count),
    )
    retrieval_time = time.perf_counter() - t1

    # Build context
    matches = []
    context_parts = []
    for i in range(len(results["documents"][0])):
        source = results["metadatas"][0][i]["source"]
        text = results["documents"][0][i]
        distance = results["distances"][0][i]
        matches.append({"source": source, "text": text, "distance": distance})
        context_parts.append(f"[{i+1}] (Source: {source})\n{text}")
    context = "\n\n".join(context_parts)
    system_prompt = SYSTEM_TEMPLATE.format(context=context)

    # Step 3: Generate answer
    t2 = time.perf_counter()
    answer, chat_usage = chat_completion(system_prompt, question, chat_model)
    generation_time = time.perf_counter() - t2

    total_time = embed_time + retrieval_time + generation_time

    metrics = {
        "question": question,
        "answer": answer,
        "sources": [m["source"] for m in matches],
        "distances": [m["distance"] for m in matches],
        # Latency (seconds)
        "embed_latency_s": round(embed_time, 3),
        "retrieval_latency_s": round(retrieval_time, 3),
        "generation_latency_s": round(generation_time, 3),
        "total_latency_s": round(total_time, 3),
        # Token usage
        "embed_tokens": embed_usage.get("total_tokens", 0),
        "chat_prompt_tokens": chat_usage.get("prompt_tokens", 0),
        "chat_completion_tokens": chat_usage.get("completion_tokens", 0),
        "chat_total_tokens": chat_usage.get("total_tokens", 0),
    }
    return metrics


def estimate_cost(chat_model: str, embed_tokens: int, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a single query."""
    embed_price = PRICING_PER_1M_TOKENS.get("mistral-embed", {"input": 0, "output": 0})
    chat_price = PRICING_PER_1M_TOKENS.get(chat_model, {"input": 0, "output": 0})

    cost = (
        embed_tokens * embed_price["input"] / 1_000_000
        + prompt_tokens * chat_price["input"] / 1_000_000
        + completion_tokens * chat_price["output"] / 1_000_000
    )
    return cost


# ═══════════════════════════════════════════════════════════════
#  Main benchmark loop
# ═══════════════════════════════════════════════════════════════

def run_benchmark():
    if not MISTRAL_API_KEY:
        print("ERROR: MISTRAL_API_KEY is not set. Check your .env file.")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Temporary ChromaDB for benchmark collections
    bench_chroma_dir = os.path.join(RESULTS_DIR, "chroma_bench")
    chroma_client = chromadb.PersistentClient(path=bench_chroma_dir)

    all_results = []

    print("=" * 60)
    print("  RAG Benchmark Runner")
    print("=" * 60)
    print(f"  Questions : {len(BENCHMARK_QUESTIONS)}")
    print(f"  Configs   : {len(BENCHMARK_CONFIGS)}")
    print(f"  Documents : {DOCUMENTS_DIR}")
    print("=" * 60)

    for cfg in BENCHMARK_CONFIGS:
        name = cfg["name"]
        top_k = cfg["top_k"]
        chunk_size = cfg["chunk_size"]
        chunk_overlap = cfg["chunk_overlap"]
        chat_model = cfg["chat_model"]

        print(f"\n▸ {name}: model={chat_model}, top_k={top_k}, "
              f"chunk_size={chunk_size}, overlap={chunk_overlap}")

        # Build / reuse the chunked collection
        col_name = build_collection(chunk_size, chunk_overlap, chroma_client)
        collection = chroma_client.get_collection(col_name)

        for qi, question in enumerate(BENCHMARK_QUESTIONS, 1):
            print(f"  Q{qi}: {question[:60]}...")
            metrics = run_query(question, collection, top_k, chat_model)

            if "error" in metrics:
                print(f"      ⚠  {metrics['error']}")
                continue

            cost = estimate_cost(
                chat_model,
                metrics["embed_tokens"],
                metrics["chat_prompt_tokens"],
                metrics["chat_completion_tokens"],
            )

            result = {
                "config_name": name,
                "chat_model": chat_model,
                "top_k": top_k,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "question_index": qi,
                **metrics,
                "estimated_cost_usd": round(cost, 6),
            }
            all_results.append(result)

            print(f"      latency={metrics['total_latency_s']}s  "
                  f"tokens={metrics['chat_total_tokens']}  "
                  f"cost=${cost:.6f}")

    # ── Save detailed JSON ────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Detailed results → {json_path}")

    # ── Save summary CSV ──────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, f"benchmark_{timestamp}.csv")
    csv_fields = [
        "config_name", "chat_model", "top_k", "chunk_size", "chunk_overlap",
        "question_index", "question",
        "total_latency_s", "embed_latency_s", "retrieval_latency_s", "generation_latency_s",
        "embed_tokens", "chat_prompt_tokens", "chat_completion_tokens", "chat_total_tokens",
        "estimated_cost_usd",
        "answer",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"✓ Summary CSV      → {csv_path}")

    # ── Print aggregate table ─────────────────────────────────
    print("\n" + "=" * 80)
    print("  AGGREGATE RESULTS (averages per config)")
    print("=" * 80)
    print(f"{'Config':<14} {'Model':<24} {'top_k':>5} {'chunk':>6} "
          f"{'Avg Latency':>12} {'Avg Tokens':>11} {'Total Cost':>11}")
    print("-" * 80)

    from itertools import groupby
    from operator import itemgetter

    for cfg_name, group in groupby(all_results, key=itemgetter("config_name")):
        items = list(group)
        n = len(items)
        avg_lat = sum(r["total_latency_s"] for r in items) / n
        avg_tok = sum(r["chat_total_tokens"] for r in items) / n
        total_cost = sum(r["estimated_cost_usd"] for r in items)
        model = items[0]["chat_model"]
        top_k = items[0]["top_k"]
        chunk = items[0]["chunk_size"]
        print(f"{cfg_name:<14} {model:<24} {top_k:>5} {chunk:>6} "
              f"{avg_lat:>10.3f}s {avg_tok:>10.0f} ${total_cost:>9.6f}")

    print()

    # ── Generate survey HTML ──────────────────────────────────
    survey_path = os.path.join(RESULTS_DIR, f"survey_{timestamp}.html")
    generate_survey_html(all_results, survey_path)
    print(f"✓ Survey HTML      → {survey_path}")
    print("\nDone! Open the survey HTML in a browser for quality evaluation.\n")


# ═══════════════════════════════════════════════════════════════
#  Survey HTML generator
# ═══════════════════════════════════════════════════════════════

def generate_survey_html(results: list[dict], output_path: str):
    """
    Generate an HTML page that shows each question with all config answers
    side by side, with radio buttons for rating (1-5).
    The config names are hidden (shown as Config A, B, C...) to avoid bias.
    Results can be exported as JSON from the browser.
    """
    from itertools import groupby
    from operator import itemgetter

    # Group results by question
    questions = {}
    for r in results:
        qi = r["question_index"]
        if qi not in questions:
            questions[qi] = {"question": r["question"], "answers": []}
        questions[qi]["answers"].append({
            "config_name": r["config_name"],
            "answer": r["answer"],
        })

    # Shuffle labels to blind the evaluator
    config_names = list(dict.fromkeys(r["config_name"] for r in results))
    labels = [chr(65 + i) for i in range(len(config_names))]  # A, B, C, ...
    label_map = dict(zip(config_names, labels))

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Benchmark — Quality Survey</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f5f5; padding: 2rem; color: #333; }
  h1 { text-align: center; margin-bottom: .5rem; }
  .subtitle { text-align: center; color: #666; margin-bottom: 2rem; }
  .question-block { background: #fff; border-radius: 12px; padding: 1.5rem; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,.08); }
  .question-block h2 { color: #1a73e8; margin-bottom: 1rem; font-size: 1.1rem; }
  .question-text { background: #e8f0fe; padding: .75rem 1rem; border-radius: 8px; margin-bottom: 1rem; font-weight: 500; }
  .answers-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
  .answer-card { border: 2px solid #e0e0e0; border-radius: 8px; padding: 1rem; }
  .answer-card h3 { margin-bottom: .5rem; color: #444; }
  .answer-card .answer-text { font-size: .9rem; line-height: 1.5; white-space: pre-wrap; max-height: 300px; overflow-y: auto; margin-bottom: .75rem; }
  .rating { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
  .rating label { cursor: pointer; padding: .25rem .6rem; border: 1px solid #ccc; border-radius: 4px; font-size: .85rem; }
  .rating input[type="radio"] { display: none; }
  .rating input[type="radio"]:checked + span { background: #1a73e8; color: #fff; border-color: #1a73e8; padding: .25rem .6rem; border-radius: 4px; }
  .export-section { text-align: center; margin: 2rem 0; }
  .export-btn { background: #1a73e8; color: #fff; border: none; padding: .75rem 2rem; border-radius: 8px; font-size: 1rem; cursor: pointer; }
  .export-btn:hover { background: #1557b0; }
  .legend { background: #fff3cd; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; }
  .legend p { margin: .25rem 0; font-size: .9rem; }
</style>
</head>
<body>
<h1>RAG Benchmark — Quality Survey</h1>
<p class="subtitle">Rate each answer from 1 (poor) to 5 (excellent). Config identities are hidden.</p>

<div class="legend">
  <p><strong>Rating Guide:</strong></p>
  <p>1 = Incorrect / irrelevant &nbsp; 2 = Partially correct but missing key info &nbsp; 3 = Acceptable answer</p>
  <p>4 = Good, mostly complete &nbsp; 5 = Excellent, accurate and well-cited</p>
</div>
"""

    for qi in sorted(questions.keys()):
        q = questions[qi]
        html += f'<div class="question-block">\n'
        html += f'  <h2>Question {qi}</h2>\n'
        html += f'  <div class="question-text">{_escape_html(q["question"])}</div>\n'
        html += f'  <div class="answers-grid">\n'

        for ans in q["answers"]:
            label = label_map[ans["config_name"]]
            field_name = f"q{qi}_{label}"
            html += f'    <div class="answer-card">\n'
            html += f'      <h3>Version {label}</h3>\n'
            html += f'      <div class="answer-text">{_escape_html(ans["answer"])}</div>\n'
            html += f'      <div class="rating">\n'
            html += f'        <span>Rating:</span>\n'
            for score in range(1, 6):
                html += (f'        <label><input type="radio" name="{field_name}" '
                         f'value="{score}"><span>{score}</span></label>\n')
            html += f'      </div>\n'
            html += f'    </div>\n'

        html += f'  </div>\n'
        html += f'</div>\n'

    # Config mapping (hidden at bottom, revealed on export)
    mapping_json = json.dumps(dict(zip(labels, config_names)))

    html += """
<div class="export-section">
  <button class="export-btn" onclick="exportResults()">Export Ratings as JSON</button>
</div>

<script>
const CONFIG_MAP = """ + mapping_json + """;

function exportResults() {
  const radios = document.querySelectorAll('input[type="radio"]:checked');
  const ratings = {};
  radios.forEach(r => {
    ratings[r.name] = parseInt(r.value);
  });

  const output = {
    timestamp: new Date().toISOString(),
    config_mapping: CONFIG_MAP,
    ratings: ratings
  };

  const blob = new Blob([JSON.stringify(output, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'survey_ratings.json';
  a.click();
  URL.revokeObjectURL(url);
}
</script>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def _escape_html(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("\n", "<br>"))


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_benchmark()
