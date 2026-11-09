import json
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PAGES = Path("processed/pages.json")
OUT = Path("processed/summaries.json")
MODEL = "sshleifer/distilbart-cnn-12-6"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()


def summarize_text(text, max_len=120, min_len=30):
    enc = tok(text, return_tensors="pt", truncation=True,
              max_length=1000)
    with torch.no_grad():
        out = model.generate(**enc, max_length=max_len,
                             min_length=min_len, num_beams=2)
    return tok.decode(out[0], skip_special_tokens=True)


def split_words(text, size=600):
    w = text.split()
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]


def summarize_long(text):
    parts = [summarize_text(c) for c in split_words(text)]
    while len(parts) > 1:
        joined = " ".join(parts)
        parts = [summarize_text(c) for c in split_words(joined)]
    return parts[0] if parts else ""


def summarize_chapters():
    pages = json.loads(PAGES.read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for p in pages:
        groups[(p["file"], p["chapter"])].append(p)

    results = []
    for (file, chapter), ps in groups.items():
        text = " ".join(p["text"] for p in ps)
        if len(text.split()) < 50:
            continue
        final = summarize_long(text)
        pr = f"{ps[0]['page']}-{ps[-1]['page']}"
        results.append({"file": file, "chapter": chapter,
                        "pages": pr, "summary": final})
        print(f"\n[{file}] {chapter} (p.{pr})")
        print(final)

    OUT.write_text(json.dumps(results), encoding="utf-8")
    print("\nSaved processed/summaries.json")


if __name__ == "__main__":
    summarize_chapters()