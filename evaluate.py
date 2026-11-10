"""Scores eval/questions.json three ways:
- retrieval: for answerable questions, is the expected source among the retrieved chunks?
- citations: for answerable questions, does the answer contain the expected words AND cite the
  expected source?
- refusals: for questions the documents do not answer, does the system refuse?
python evaluate.py [--fake]"""

import json
import os
import sys

from docqa.index import BM25Index, load_chunks
from docqa.qa import REFUSAL, FakeLLM, GroqLLM, answer, citations


def score(questions, index, llm):
    rows = []
    for q in questions:
        reply, hits = answer(q["question"], index, llm)
        retrieved = {c.source for _, c in index.search(q["question"])}
        if q["answer_contains"] is None:
            ok = reply == REFUSAL
            kind, retrieval_ok = "refusal", None
        else:
            ok = q["answer_contains"].lower() in reply.lower() and q["source"] in citations(reply)
            kind, retrieval_ok = "answer", q["source"] in retrieved
        rows.append({"id": q["id"], "kind": kind, "retrieval_ok": retrieval_ok, "ok": ok, "reply": reply})
    return rows


def summary(rows):
    answerable = [r for r in rows if r["kind"] == "answer"]
    refusals = [r for r in rows if r["kind"] == "refusal"]
    return {"retrieval": (sum(r["retrieval_ok"] for r in answerable), len(answerable)),
            "citations": (sum(r["ok"] for r in answerable), len(answerable)),
            "refusals": (sum(r["ok"] for r in refusals), len(refusals))}


def main():
    fake = "--fake" in sys.argv or not os.environ.get("GROQ_API_KEY")
    llm = FakeLLM() if fake else GroqLLM()
    rows = score(json.load(open("eval/questions.json")), BM25Index(load_chunks("docs")), llm)
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), r["id"], r["reply"][:110])
    total = sum(r["ok"] for r in rows)
    scores = summary(rows)
    print(f"\n{type(llm).__name__}: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in scores.items())
          + f"; overall {total}/{len(rows)}")
    os.makedirs("results", exist_ok=True)
    json.dump({"model": type(llm).__name__, "scores": scores, "passed": total, "total": len(rows), "rows": rows},
              open(f"results/eval_{'fake' if fake else 'groq'}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
