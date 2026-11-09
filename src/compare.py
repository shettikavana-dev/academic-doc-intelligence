import re
import sys
from collections import defaultdict

from index import search  # keep your original import line here if it differs


def clean(text, limit=350):
    """Tidy PDF text, then cut at a sentence or word boundary."""
    text = re.sub(r"(\w)- (\w)", r"\1\2", text)        # fash- ion -> fashion
    text = re.sub(r"[\u25a1\u2610\uf0a7\uf0b7\u2022\ufffd]", "•", text)  # bullet glyphs
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if end > limit // 2:
        return cut[: end + 1]
    return cut.rsplit(" ", 1)[0] + "..."


def compare(topic, per_doc=2, k=100, min_score=0.30):
    """Return {file: [(score, chunk), ...]} with up to per_doc chunks per file.

    Chunks scoring below min_score are skipped, and only one chunk
    per (file, page) is kept.
    """
    hits = search(topic, k=k)
    groups = defaultdict(list)
    seen = set()
    for score, c in hits:
        if score < min_score:
            continue
        key = (c["file"], c["page"])
        if key in seen:
            continue
        if len(groups[c["file"]]) < per_doc:
            groups[c["file"]].append((score, c))
            seen.add(key)
    return groups


if __name__ == "__main__":
    topic = input("Topic to compare: ").strip()
    if not topic:
        sys.exit("Please enter a topic.")

    results = compare(topic)
    if not results:
        sys.exit("No results found.")

    for file, items in results.items():
        print("\n" + "=" * 60)
        print(file)
        for score, c in items:
            print(f"\n[p.{c['page']}, {c.get('type', 'text')}] score={score:.2f}")
            print(clean(c["text"]))