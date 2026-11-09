import json, re
from pathlib import Path
import pymupdf

DATA_DIR = Path("data")
OUT_FILE = Path("processed/pages.json")
KNOWN_TYPES = {"textbooks", "notes", "papers"}


def guess_type(path):
    p = path.parent.name.lower()
    return p if p in KNOWN_TYPES else "document"


def ingest_pdf(path):
    doc = pymupdf.open(path)
    toc = sorted((pg, t) for lvl, t, pg in doc.get_toc() if lvl == 1)
    current = "Unknown"
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue
        if toc:
            for start, title in toc:
                if start <= i:
                    current = title
        else:
            m = re.search(r"^\s*(chapter|unit|module)\s+\d+.*$",
                          text, re.I | re.M)
            if m:
                current = m.group(0).strip()[:80]
        pages.append({
            "file": path.name,
            "type": guess_type(path),
            "page": i,
            "chapter": current,
            "text": text,
        })
    return pages


def ingest_all(folder=DATA_DIR):
    all_pages = []
    for pdf in sorted(folder.rglob("*.pdf")):
        pages = ingest_pdf(pdf)
        print(f"{pdf.name}: {len(pages)} pages")
        all_pages.extend(pages)
    OUT_FILE.parent.mkdir(exist_ok=True)
    OUT_FILE.write_text(json.dumps(all_pages), encoding="utf-8")
    print(f"Saved {len(all_pages)} pages")


if __name__ == "__main__":
    ingest_all()