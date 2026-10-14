# NoteMind AI — Local Search Upgrade

This upgrade adds useful local commands to the existing NoteMind AI starter app.

## New feature: local mode

Run from the NoteMind AI project folder:

```bash
python app.py --local
```

Local mode does **not** need a Gemini API key or internet connection. It reads `.txt` and `.md` files from `docs/` on this computer only.

Commands:

- `notes` — list local note files
- `stats` — show file, line, word, and size counts
- `find Fourier transform` — case-insensitive phrase search through note lines
- `help` — show commands
- `exit` — quit

The normal `python app.py` RAG mode is still available. In RAG mode, notes are sent to Google's Gemini API during indexing and relevant excerpts are sent when answering questions. Do not use private or sensitive notes unless you are comfortable with that data handling.

## Other improvements in this version

- Reads `GEMINI_API_KEY` from the environment, with backward compatibility for an existing local `secret.py` file.
- Sends the API key in a request header instead of putting it in the URL.
- Adds explicit request timeouts and clearer API errors.
- Adds `:notes`, `:stats`, and `:find` commands to RAG mode too. These commands use the local files/index; `:find` does not call Gemini.
- Uses retrieval-specific embedding task types for note chunks and questions.
- Makes paths relative to `app.py`, so running the script from another working directory is less error-prone.
- Validates the local vector-store structure before using it.

## Setup

Install the dependencies already used by the app:

```bash
python -m pip install numpy requests
```

Place `.txt` or `.md` files inside `docs/`.

For Gemini mode, set the API key in the VS Code terminal (PowerShell):

```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
python app.py
```

That command sets the key for the current terminal session only. To persist the variable across new terminals, use Windows' environment-variable settings, or `setx GEMINI_API_KEY "YOUR_KEY_HERE"` and then open a new terminal.

Keep the following local files out of GitHub:

```gitignore
.env
secret.py
vector_store.json
docs/
__pycache__/
```

## Suggested tests

1. Run `python app.py --local` with no API key configured.
2. Use `notes`, `stats`, and `find <a word in your notes>`.
3. Confirm local search works with Wi-Fi turned off if you want to verify that it is offline.
4. Run `python app.py` only after configuring a Gemini API key.
5. Review `.gitignore` before committing; never commit your key, original notes, or vector store.
