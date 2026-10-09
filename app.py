"""
NOTEMIND AI — STARTER KIT (v1)
====================================
Your first RAG (Retrieval-Augmented Generation) app — an AI that answers
questions from YOUR notes, with sources. Free to run: it uses Google's
free Gemini API, so the heavy thinking happens on Google's servers,
not your laptop. Works even on old/slow machines!

WHAT THIS DOES
--------------
1. You drop your college notes (.txt or .md files) into a folder called `docs`
2. Run this file once — it reads the notes, chops them into small chunks,
   and turns each chunk into a list of numbers (an "embedding" = meaning
   turned into numbers)
3. You ask questions in the terminal — it finds the most relevant chunks
   and asks Google's Gemini AI (running on Google's servers, NOT your
   laptop) to answer based on YOUR notes, with sources. Open-book exam
   for the AI. 😎  This works even on old/slow laptops!

SETUP (do this once)
--------------------
1. Install Python 3.10+ from python.org (tick "Add Python to PATH")
2. Get your FREE Gemini API key (takes 2 minutes):
   - Go to https://aistudio.google.com and sign in with your Google account
   - Click "Get API key" -> "Create API key", then copy it
3. PRIVACY! Keep the key OUT of app.py (never hardcode it):
   - Windows terminal:  setx GEMINI_API_KEY "your-key-here"
     (then close and REOPEN VS Code so it takes effect)
   - Mac/Linux:         export GEMINI_API_KEY="your-key-here"
   - This code reads the key from that environment variable
4. In this project folder, install the 2 libraries we need:
       pip install requests numpy
5. Create a folder named `docs` next to this file and drop your notes in it
6. Run:
       python app.py

PRIVACY NOTES (read these, seriously)
------------------------------------
- Your API key lives ONLY in the environment variable, never inside this
  file. If it ever leaks, delete it in AI Studio and create a fresh one.
- vector_store.json contains your actual notes -> keep it out of GitHub.
- The text of your notes IS sent to Google's servers to get AI answers
  (that's how it runs on any laptop). On the free tier, Google's terms
  allow them to use your requests to improve their products — fine for
  study notes, but NEVER feed it passwords, IDs, or personal documents.
- Before pushing to GitHub, create a .gitignore file in this folder with
  exactly these lines:
       .env
       vector_store.json
       docs/
       __pycache__/
"""

import json
import os
from pathlib import Path
import time
import numpy as np
import requests

# ------------------------- SETTINGS (tweak me!) -------------------------
# PRIVACY: the API key is read from an environment variable, so it never
# sits inside this file (and this file is what goes to GitHub).
#   Windows:    setx GEMINI_API_KEY "your-key-here"   (then restart VS Code)
#   Mac/Linux:  export GEMINI_API_KEY="your-key-here"
from secret import API_KEY
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
LLM_MODEL = "gemini-flash-latest"           # fast, free tier, runs on Google's servers
EMBED_MODEL = "gemini-embedding-001"     # turns text into numbers
DOCS_DIR = Path("docs")                 # folder with your notes
STORE_FILE = Path("vector_store.json")  # local "database" (just a file!)
CHUNK_SIZE = 800                        # characters per chunk of a note
TOP_K = 5                               # how many chunks to hand the LLM

# ------------------- TALKING TO GEMINI (two functions) -------------------


def check_api_key():
    """Friendly crash if the GEMINI_API_KEY environment variable isn't set."""
    if not API_KEY:
        raise SystemExit(
            "\n⚠️  GEMINI_API_KEY is not set! Open a terminal and run:\n\n"
            "     Windows:    setx GEMINI_API_KEY \"your-key-here\"\n"
            "     Mac/Linux:  export GEMINI_API_KEY=\"your-key-here\"\n\n"
            "Then close and REOPEN VS Code, and run me again.\n"
            "(Get your free key at https://aistudio.google.com)\n"
        )


def gemini_embed(texts):
    """Send a list of texts to Gemini, get back a list of embeddings."""
    response = requests.post(
        f"{GEMINI_URL}/models/{EMBED_MODEL}:batchEmbedContents",
        params={"key": API_KEY},
        json={
            "requests": [
                {
                    "model": f"models/{EMBED_MODEL}",
                    "content": {"parts": [{"text": t}]},
                }
                for t in texts
            ]
        },
    )
    response.raise_for_status()  # crash loudly if the key or internet fails
    return [np.array(e["values"], dtype=float) for e in response.json()["embeddings"]]


MODELS = ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"]

def gemini_generate(prompt):
    """Try each model in turn; switch if one is busy or missing."""
    for round_no in range(2):
        for model in MODELS:
            response = requests.post(
                f"{GEMINI_URL}/models/{model}:generateContent",
                params={"key": API_KEY},
                json={"contents": [{"parts": [{"text": prompt}]}]},
            )
            if response.status_code in (404, 429, 503):
                print(f"  {model} unavailable ({response.status_code}), trying next...")
                continue
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        time.sleep(10)
    raise requests.exceptions.HTTPError("All models busy, try again in a minute")


# ----------------------- STEP 1: CHUNKING THE NOTES -----------------------


def chunk_text(text):
    """Split a long note into bite-sized chunks the AI can search.

    Simple strategy: split on blank lines (paragraphs), then pack
    paragraphs together until we reach CHUNK_SIZE characters.
    """
    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        # A single huge paragraph? Slice it hard by size.
        while len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(paragraph[:CHUNK_SIZE].strip())
            paragraph = paragraph[CHUNK_SIZE:]
        if len(current) + len(paragraph) <= CHUNK_SIZE:
            current += "\n\n" + paragraph
        else:
            if current.strip():
                chunks.append(current.strip())
            current = paragraph
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ------------------------ STEP 2: BUILDING THE STORE ------------------------
def add_overlap(chunks, n=150):
    return [chunks[0]] + [chunks[i-1][-n:] + "\n" + chunks[i] for i in range(1, len(chunks))] if chunks else chunks

def ingest():
    """Read every note in docs/, chunk it, embed it, save to vector_store.json."""
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir()
        print(f"Created the '{DOCS_DIR}' folder — now drop your .txt notes in it and run me again!")
        return False

    files = [p for p in DOCS_DIR.iterdir() if p.suffix.lower() in (".txt", ".md")]
    if not files:
        print(f"No .txt or .md files found in '{DOCS_DIR}/'. Add some notes and run again.")
        return False

    all_chunks, all_sources = [], []
    for path in files:
        print(f"  Reading {path.name} ...")
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in add_overlap(chunk_text(text)):
            all_chunks.append(chunk)
            all_sources.append(path.name)

    print(f"  Made {len(all_chunks)} chunks. Embedding them (a few seconds)...")
    # Gemini accepts up to 100 texts per embedding request, so send in batches
    embeddings = []
    for i in range(0, len(all_chunks), 50):
        batch = all_chunks[i : i + 50]
        print(f"  ... embedding batch {i // 50 + 1} ({len(batch)} chunks)")
        embeddings.extend(gemini_embed(batch))

    store = {
        "chunks": all_chunks,
        "sources": all_sources,
        "embeddings": [e.tolist() for e in embeddings],
    }
    STORE_FILE.write_text(json.dumps(store))
    print(f"  Saved to {STORE_FILE}. Your notes are now searchable! 🎉")
    return True


def load_store():
    """Load the saved chunks + embeddings from disk into memory."""
    store = json.loads(STORE_FILE.read_text())
    matrix = np.array(store["embeddings"])                 # shape: (n_chunks, dim)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # length of each row
    matrix = matrix / np.clip(norms, 1e-10, None)         # normalize -> cosine similarity
    return store["chunks"], store["sources"], matrix


# ------------------------ STEP 3: RETRIEVAL (search) ------------------------


def retrieve(question, chunks, matrix, k=TOP_K):
    """Find the k chunks whose meaning is closest to the question."""
    question_vec = gemini_embed([question])[0]
    question_vec = question_vec / np.linalg.norm(question_vec)  # normalize too
    scores = matrix @ question_vec  # cosine similarity vs every chunk
    best = np.argsort(scores)[::-1][:k]  # top-k indexes, highest score first
    return best


# ------------------------- STEP 4: THE FULL PIPELINE -------------------------


def answer(question, chunks, sources, matrix):
    """Retrieve relevant chunks, then ask the LLM to answer using them."""
    best = retrieve(question, chunks, matrix)
    context = "\n\n---\n\n".join(
        f"[Source: {sources[i]}]\n{chunks[i]}" for i in best
    )
    prompt = (
        "You are a helpful study assistant. Answer the student's question "
        "using ONLY the context below. Mention which source you used. "
        "If the context doesn't contain the answer, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )
    reply = gemini_generate(prompt)
    cited = sorted({sources[i] for i in best})
    return reply, cited, best


# --------------------------------- MAIN ---------------------------------


if __name__ == "__main__":
    print("=" * 60)
    print("  NOTEMIND AI  --  your notes + Google Gemini  🚀")
    print("=" * 60)
    check_api_key()

    # First run ever? Build the store. Deleted or changed your notes?
    # Delete vector_store.json and run again (or type 'rebuild').
    if not STORE_FILE.exists():
        if not ingest():
            raise SystemExit(1)

    chunks, sources, matrix = load_store()
    print(f"\nLoaded {len(chunks)} chunks. Ask me anything about your notes!")
    print("(Type 'exit' to quit, 'rebuild' to re-read the docs folder)\n")

    while True:
        question = input("You > ").strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Bye! Come back with more notes 📚")
            break
        if question.lower() == "rebuild":
            if ingest():
                chunks, sources, matrix = load_store()
            continue
        try:
            reply, cited, best = answer(question, chunks, sources, matrix)
            print(f"\nAI > {reply}")
            for i in best:
                print(f"   📄 {sources[i]}: {chunks[i][:100].replace("\n", " ").strip()}...")
            print()
        except requests.exceptions.HTTPError as err:
            print(f"\n⚠️  Gemini returned an error: {err.response.status_code if err.response is not None else 'busy'}. "
                  "Double-check your API key (in SETTINGS) and your internet.\n")
        except requests.exceptions.ConnectionError:
            print("\n⚠️  Can't reach the internet! Check your connection and retry.\n")