import torch
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from index import search

RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QA_MODEL = "deepset/roberta-base-squad2"

reranker = CrossEncoder(RERANK)
tok = AutoTokenizer.from_pretrained(QA_MODEL)
qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).eval()


def extract(question, context, max_len=30):
    enc = tok(question, context, return_tensors="pt",
              truncation="only_second", max_length=384,
              return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0]
    seq = enc.sequence_ids(0)
    with torch.no_grad():
        out = qa_model(**enc)
    s = out.start_logits[0].softmax(-1)
    e = out.end_logits[0].softmax(-1)
    mask = torch.tensor([x == 1 for x in seq])
    s = s * mask
    e = e * mask
    m = s[:, None] * e[None, :]
    m = torch.triu(m) - torch.triu(m, diagonal=max_len)
    idx = int(m.argmax())
    i, j = divmod(idx, m.shape[1])
    text = context[int(offsets[i][0]):int(offsets[j][1])]
    return text, float(m[i, j])


def answer(question, k=20, top=3):
    hits = search(question, k=k)
    pairs = [(question, c["text"]) for _, c in hits]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, hits), key=lambda x: -x[0])[:top]

    results = []
    for _, (_, c) in ranked:
        text, conf = extract(question, c["text"])
        results.append({
            "answer": text,
            "confidence": conf,
            "file": c["file"],
            "page": c["page"],
            "context": c["text"],
        })
    results.sort(key=lambda r: -r["confidence"])
    return results


if __name__ == "__main__":
    q = input("Ask a question: ")
    for r in answer(q):
        print(f"\nAnswer: {r['answer']}")
        print(f"Source: [{r['file']}, p.{r['page']}]  "
              f"confidence={r['confidence']:.2f}")