import base64, io, time
from pathlib import Path
import requests
from pypdf import PdfReader, PdfWriter
from secret import API_KEY

MODEL = "gemini-flash-latest"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
PROMPT = "Transcribe all text from this document exactly as written, as plain text. Keep headings and formulas. No commentary."
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def ocr(data, mime):
    body = {"contents": [{"parts": [
        {"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode()}},
        {"text": PROMPT}]}]}
    for i in range(5):
        r = requests.post(URL, params={"key": API_KEY}, json=body, timeout=180)
        if r.status_code in (429, 503):
            print(f"  busy ({r.status_code}), waiting {20 * (i + 1)}s...")
            time.sleep(20 * (i + 1))
            continue
        if not r.ok:
            raise SystemExit(f"Error {r.status_code}: {r.text[:200]}")
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)
    raise SystemExit("Still busy. Wait a while (or until tomorrow) and rerun.")


def pdf_to_text(f):
    reader = PdfReader(f)
    out, scanned = {}, []
    for i, pg in enumerate(reader.pages):
        t = (pg.extract_text() or "").strip()
        if len(t) > 100:
            out[i] = t          # typed page, free
        else:
            scanned.append(i)   # needs OCR
    print(f"{f.name}: {len(out)} typed pages, {len(scanned)} scanned pages")
    for s in range(0, len(scanned), 3):
        group = scanned[s:s + 3]
        w = PdfWriter()
        for i in group:
            w.add_page(reader.pages[i])
        buf = io.BytesIO()
        w.write(buf)
        print(f"  OCR pages {[i + 1 for i in group]}")
        out[group[0]] = ocr(buf.getvalue(), "application/pdf")
        time.sleep(8)
    return "\n\n".join(out[k] for k in sorted(out))


Path("docs").mkdir(exist_ok=True)
for f in Path("scans").iterdir():
    ext = f.suffix.lower()
    out = Path("docs") / (f.stem + ".txt")
    if out.exists():
        continue
    if ext == ".pdf":
        text = pdf_to_text(f)
    elif ext in MIME:
        text = ocr(f.read_bytes(), MIME[ext])
    else:
        continue
    out.write_text(text, encoding="utf-8")
    print(f"Saved {out}")