"""NoteMind AI - a lightweight notes RAG assistant.

Normal mode (uses Gemini over the internet):
    python app.py

Local mode (searches files on this computer; no API key or internet needed):
    python app.py --local

Supported note files: .txt and .md inside the docs/ directory beside this file.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
STORE_FILE = BASE_DIR / "vector_store.json"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
LLM_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]
EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
REQUEST_TIMEOUT = 60


def get_api_key():
    """Read the key from the environment, with a backward-compatible secret.py fallback."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        return key

    # Existing local copies may already use secret.py. Never commit that file.
    try:
        from secret import API_KEY  # type: ignore

        return str(API_KEY).strip()
    except (ImportError, AttributeError):
        return ""


API_KEY = get_api_key()


def api_headers():
    if not API_KEY:
        raise RuntimeError(
            "Gemini API key is missing. Set GEMINI_API_KEY or add a local secret.py."
        )
    return {"x-goog-api-key": API_KEY, "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Local-only note browser and keyword search
# ---------------------------------------------------------------------------


def get_note_files():
    """Return .txt and .md files directly inside docs/, in stable name order."""
    if not DOCS_DIR.exists():
        return []
    return sorted(
        (path for path in DOCS_DIR.iterdir()
         if path.is_file() and path.suffix.lower() in {".txt", ".md"}),
        key=lambda path: path.name.casefold(),
    )


def find_local_matches(query, limit=10):
    """Search note lines locally without calling Gemini or reading vector_store.json."""
    query = query.strip()
    if not query:
        return []

    query_lower = query.casefold()
    matches = []
    for path in get_note_files():
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"Could not read {path.name}: {exc}")
            continue

        for line_number, line in enumerate(lines, start=1):
            if query_lower in line.casefold():
                clean_line = " ".join(line.split())
                if len(clean_line) > 220:
                    position = clean_line.casefold().find(query_lower)
                    start = max(0, position - 70)
                    end = min(len(clean_line), position + len(query) + 110)
                    clean_line = ("…" if start else "") + clean_line[start:end]
                    if end < len(" ".join(line.split())):
                        clean_line += "…"
                matches.append(
                    {"source": path.name, "line": line_number, "text": clean_line}
                )
                if len(matches) >= limit:
                    return matches
    return matches


def print_local_help():
    print(
        "\nLocal commands (no API key or internet needed):\n"
        "  notes              List available .txt and .md files\n"
        "  stats              Show local note/file/word counts\n"
        "  find <text>        Search note text (case-insensitive)\n"
        "  help               Show these commands\n"
        "  exit               Quit local mode\n"
        "\nExample: find Fourier transform\n"
    )


def run_local_mode():
    """Interactive local mode. It only reads files from docs/ and makes no API calls."""
    print("=" * 58)
    print("  NOTEMIND AI — LOCAL NOTES TOOLS")
    print("  Offline keyword search; note text stays on this computer.")
    print("=" * 58)

    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created {DOCS_DIR}. Add .txt or .md notes and start local mode again.")

    print_local_help()
    while True:
        try:
            command = input("Local > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting local mode.")
            break

        if not command:
            continue

        action, _, argument = command.partition(" ")
        action = action.casefold()
        argument = argument.strip()

        if action in {"exit", "quit"}:
            print("Exiting local mode.")
            break
        if action in {"help", ":help", "/help"}:
            print_local_help()
        elif action in {"notes", ":notes", "/notes"}:
            files = get_note_files()
            if not files:
                print(f"No .txt or .md notes found in {DOCS_DIR}.")
                continue
            print(f"\nFound {len(files)} note file(s):")
            for path in files:
                print(f"  • {path.name} ({path.stat().st_size:,} bytes)")
            print()
        elif action in {"stats", ":stats", "/stats"}:
            files = get_note_files()
            total_lines = 0
            total_words = 0
            total_bytes = 0
            for path in files:
                try:
                    content = path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    print(f"Could not read {path.name}: {exc}")
                    continue
                total_lines += len(content.splitlines())
                total_words += len(content.split())
                total_bytes += path.stat().st_size
            print("\nLocal note statistics")
            print(f"  Files: {len(files)}")
            print(f"  Lines: {total_lines:,}")
            print(f"  Words: {total_words:,}")
            print(f"  Size:  {total_bytes:,} bytes\n")
        elif action in {"find", ":find", "/find"}:
            if not argument:
                print("Usage: find <word or phrase>")
                continue
            matches = find_local_matches(argument)
            if not matches:
                print(f"No matches for {argument!r}.\n")
                continue
            print(f"\nFound {len(matches)} matching line(s) (showing at most 10):")
            for item in matches:
                print(f"  {item['source']}:{item['line']} — {item['text']}")
            print()
        else:
            print("Unknown command. Type 'help' to see the available commands.")


# ---------------------------------------------------------------------------
# Gemini RAG mode
# ---------------------------------------------------------------------------


def chunk_text(text):
    """Split note text into chunks, preferring blank-line paragraph boundaries."""
    chunks = []
    current = ""

    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        while len(paragraph) > CHUNK_SIZE:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.append(paragraph[:CHUNK_SIZE].strip())
            paragraph = paragraph[CHUNK_SIZE:]

        if not current:
            current = paragraph
        elif len(current) + len(paragraph) + 2 <= CHUNK_SIZE:
            current += "\n\n" + paragraph
        else:
            chunks.append(current.strip())
            current = paragraph

    if current.strip():
        chunks.append(current.strip())
    return chunks


def add_overlap(chunks, overlap=CHUNK_OVERLAP):
    """Carry a short tail from each chunk into the next chunk for context continuity."""
    if not chunks:
        return []
    result = [chunks[0]]
    for index in range(1, len(chunks)):
        prefix = chunks[index - 1][-overlap:] if overlap else ""
        result.append(f"{prefix}\n{chunks[index]}" if prefix else chunks[index])
    return result


def gemini_embed(texts, task_type="RETRIEVAL_DOCUMENT"):
    """Generate embeddings in batches using Gemini's embedding endpoint."""
    if not texts:
        return []

    results = []
    for start in range(0, len(texts), 50):
        batch = texts[start:start + 50]
        payload = {
            "requests": [
                {
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                }
                for text in batch
            ]
        }
        response = requests.post(
            f"{GEMINI_URL}/models/{EMBED_MODEL}:batchEmbedContents",
            headers=api_headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        results.extend(
            np.asarray(item["values"], dtype=float)
            for item in response.json()["embeddings"]
        )
    return results


def gemini_generate(prompt):
    """Try supported Flash models in order, falling back when busy/unavailable."""
    unavailable_statuses = {404, 429, 503}
    last_response = None

    for attempt in range(2):
        for model in LLM_MODELS:
            response = requests.post(
                f"{GEMINI_URL}/models/{model}:generateContent",
                headers=api_headers(),
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code in unavailable_statuses:
                print(f"  {model} unavailable ({response.status_code}); trying the next model...")
                last_response = response
                continue
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no answer. Try rephrasing your question.")
            parts = candidates[0].get("content", {}).get("parts", [])
            answer_text = "\n".join(part.get("text", "") for part in parts).strip()
            if not answer_text:
                raise RuntimeError("Gemini returned an empty answer. Please try again.")
            return answer_text
        if attempt == 0:
            time.sleep(3)

    if last_response is not None:
        last_response.raise_for_status()
    raise RuntimeError("All configured Gemini models are busy or unavailable. Try again later.")


def ingest():
    """Read notes, create chunks and embeddings, then save a local vector store."""
    files = get_note_files()
    if not files:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"No .txt or .md files found in '{DOCS_DIR}'. Add notes and try again.")
        return False

    all_chunks = []
    all_sources = []
    for path in files:
        print(f"  Reading {path.name} ...")
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            print(f"  Skipping empty file: {path.name}")
            continue
        for chunk in add_overlap(chunk_text(text)):
            all_chunks.append(chunk)
            all_sources.append(path.name)

    if not all_chunks:
        print("The note files were empty; nothing was indexed.")
        return False

    print(f"  Created {len(all_chunks)} chunks. Generating embeddings...")
    embeddings = gemini_embed(all_chunks, task_type="RETRIEVAL_DOCUMENT")
    store = {
        "chunks": all_chunks,
        "sources": all_sources,
        "embeddings": [embedding.tolist() for embedding in embeddings],
        "embedding_model": EMBED_MODEL,
    }
    STORE_FILE.write_text(json.dumps(store), encoding="utf-8")
    print(f"  Saved the local index to {STORE_FILE.name}.")
    return True


def load_store():
    """Load the local index and normalize its vectors for cosine similarity."""
    store = json.loads(STORE_FILE.read_text(encoding="utf-8"))
    required_keys = {"chunks", "sources", "embeddings"}
    if not required_keys.issubset(store):
        raise ValueError("The vector store is missing required fields. Run 'rebuild'.")
    if not store["chunks"] or not store["embeddings"]:
        raise ValueError("The vector store is empty. Run 'rebuild' after adding notes.")
    if not (len(store["chunks"]) == len(store["sources"]) == len(store["embeddings"])):
        raise ValueError("The vector store has mismatched data. Run 'rebuild'.")

    matrix = np.asarray(store["embeddings"], dtype=float)
    if matrix.ndim != 2:
        raise ValueError("The stored embeddings are invalid. Run 'rebuild'.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-10, None)
    return store["chunks"], store["sources"], matrix


def retrieve(question, chunks, matrix, k=TOP_K):
    """Return the indices of chunks most semantically similar to a question."""
    question_vector = gemini_embed([question], task_type="RETRIEVAL_QUERY")[0]
    norm = np.linalg.norm(question_vector)
    if norm == 0:
        raise RuntimeError("Gemini returned an empty question embedding.")
    question_vector = question_vector / norm
    if matrix.shape[1] != question_vector.shape[0]:
        raise ValueError(
            "The saved vectors use a different embedding size. Delete vector_store.json and rebuild."
        )
    scores = matrix @ question_vector
    return np.argsort(scores)[::-1][:min(k, len(chunks))]


def answer(question, chunks, sources, matrix):
    """Retrieve relevant chunks and ask Gemini to answer from those sources."""
    best_indices = retrieve(question, chunks, matrix)
    context = "\n\n---\n\n".join(
        f"[Source: {sources[index]}]\n{chunks[index]}" for index in best_indices
    )
    prompt = (
        "You are a careful study assistant. Answer using only the context below. "
        "Cite the source filename in your answer. If the context does not contain "
        "the answer, say that the notes do not provide enough information.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    reply = gemini_generate(prompt)
    cited_sources = sorted({sources[index] for index in best_indices})
    return reply, cited_sources, best_indices


def print_rag_help():
    print(
        "\nCommands:\n"
        "  :help             Show commands\n"
        "  :notes            List indexed note files\n"
        "  :stats            Show index statistics\n"
        "  :find <text>      Search note text locally without using Gemini\n"
        "  rebuild           Rebuild embeddings from docs/ (uses Gemini)\n"
        "  exit              Quit\n"
        "\nFor fully local keyword searching without an API key, run: python app.py --local\n"
    )


def show_index_notes(sources):
    counts = {}
    for source in sources:
        counts[source] = counts.get(source, 0) + 1
    if not counts:
        print("No indexed notes found.")
        return
    print("\nIndexed note files:")
    for name in sorted(counts, key=str.casefold):
        print(f"  {name} — {counts[name]} chunk(s)")
    print()


def show_index_stats(chunks, sources, matrix):
    unique_sources = len(set(sources))
    embedding_dimensions = matrix.shape[1] if matrix.ndim == 2 else 0
    store_size = STORE_FILE.stat().st_size if STORE_FILE.exists() else 0
    print("\nNoteMind AI index statistics")
    print(f"  Note files: {unique_sources}")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Embedding dimensions: {embedding_dimensions}")
    print(f"  Local index size: {store_size:,} bytes\n")


def run_rag_mode():
    print("=" * 58)
    print("  NOTEMIND AI — your notes + Gemini")
    print("=" * 58)
    if not API_KEY:
        raise SystemExit(
            "Gemini API key not found. Set GEMINI_API_KEY or configure secret.py.\n"
            "For local keyword search without a key, run: python app.py --local\n"
        )

    if not STORE_FILE.exists():
        print("No index found yet; creating one from docs/...")
        if not ingest():
            raise SystemExit(1)

    try:
        chunks, sources, matrix = load_store()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Could not load {STORE_FILE.name}: {exc}\nRun 'python app.py' after deleting the index, or rebuild it.")

    print(f"Loaded {len(chunks)} chunks from {len(set(sources))} note file(s).")
    print_rag_help()

    while True:
        try:
            question = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue

        if question.casefold() in {"exit", "quit"}:
            print("Goodbye. Keep learning!")
            break
        if question.casefold() in {":help", "/help", "help"}:
            print_rag_help()
            continue
        if question.casefold() in {":notes", "/notes", "notes"}:
            show_index_notes(sources)
            continue
        if question.casefold() in {":stats", "/stats", "stats"}:
            show_index_stats(chunks, sources, matrix)
            continue
        if question.casefold().startswith((":find ", "/find ", "find ")):
            _, _, search_text = question.partition(" ")
            if search_text.startswith(":") or search_text.startswith("/"):
                _, _, search_text = search_text.partition(" ")
            matches = find_local_matches(search_text)
            if not matches:
                print(f"No local matches for {search_text!r}.\n")
            else:
                print("\nLocal matches (no Gemini request made):")
                for item in matches:
                    print(f"  {item['source']}:{item['line']} — {item['text']}")
                print()
            continue
        if question.casefold() == "rebuild":
            print("Rebuilding the index from docs/...")
            if ingest():
                chunks, sources, matrix = load_store()
                print(f"Index rebuilt: {len(chunks)} chunks from {len(set(sources))} note file(s).")
            continue

        try:
            reply, cited, best = answer(question, chunks, sources, matrix)
            print(f"\nAI > {reply}")
            print("Sources used:")
            for source in cited:
                print(f"  • {source}")
            print("Retrieved excerpts:")
            for index in best:
                snippet = " ".join(chunks[index].split())[:160]
                print(f"  • {sources[index]}: {snippet}{'…' if len(chunks[index]) > 160 else ''}")
            print()
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            body = exc.response.text[:500] if exc.response is not None else str(exc)
            print(f"\nGemini API error ({status}): {body}\n")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print("\nCould not reach Gemini. Check your internet connection and try again.\n")
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"\nCould not answer that question: {exc}\n")


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your notes with NoteMind AI.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="use local keyword search only (no API key or internet required)",
    )
    args = parser.parse_args()
    if args.local:
        run_local_mode()
    else:
        run_rag_mode()


if __name__ == "__main__":
    main()
