import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PAGES = Path("processed/pages.json")
INDEX = Path("processed/faiss.index")
CHUNKS = Path("processed/chunks.json")
MODEL = "all-MiniLM-L6-v2"


def make_chunks(pages, size=200, overlap=40):
    chunks = []
    for p in pages:
        words = p["text"].split()
        step = size - overlap
        for start in range(0, max(len(words), 1), step):
            piece = " ".join(words[start:start + size])
            if len(piece.split()) < 20:
                continue
            chunks.append({
                "file": p["file"], "type": p["type"],
                "page": p["page"], "chapter": p["chapter"],
                "text": piece,
            })
    return chunks


def build_index():
    pages = json.loads(PAGES.read_text(encoding="utf-8"))
    chunks = make_chunks(pages)
    print(f"{len(chunks)} chunks")
    model = SentenceTransformer(MODEL)
    vectors = model.encode([c["text"] for c in chunks],
                           show_progress_bar=True,
                           normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.array(vectors, dtype="float32"))
    faiss.write_index(index, str(INDEX))
    CHUNKS.write_text(json.dumps(chunks), encoding="utf-8")
    print("Index saved")


def search(query, k=5):
    model = SentenceTransformer(MODEL)
    index = faiss.read_index(str(INDEX))
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    q = model.encode([query], normalize_embeddings=True)
    scores, ids = index.search(np.array(q, dtype="float32"), k)
    return [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0])]


if __name__ == "__main__":
    build_index()
    query = input("Ask a question: ")
    for score, c in search(query):
        print(f"\n[{c['file']}, p.{c['page']}] score={score:.2f}")
        print(c["text"][:300])