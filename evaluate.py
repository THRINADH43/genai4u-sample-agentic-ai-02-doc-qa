"""Scores eval/questions.json. A question with an expected answer passes when the answer
contains the expected words AND cites the expected source. An unanswerable question passes
when the system refuses.  python evaluate.py [--fake]"""

import json
import os
import sys

from docqa.index import BM25Index, load_chunks
from docqa.qa import REFUSAL, FakeLLM, GroqLLM, answer, citations


def score(questions, index, llm):
    rows = []
    for q in questions:
        reply, _ = answer(q["question"], index, llm)
        if q["answer_contains"] is None:
            ok = reply == REFUSAL
        else:
            ok = q["answer_contains"].lower() in reply.lower() and q["source"] in citations(reply)
        rows.append({"id": q["id"], "ok": ok, "reply": reply})
    return rows


def main():
    fake = "--fake" in sys.argv or not os.environ.get("GROQ_API_KEY")
    llm = FakeLLM() if fake else GroqLLM()
    rows = score(json.load(open("eval/questions.json")), BM25Index(load_chunks("docs")), llm)
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), r["id"], r["reply"][:110])
    total = sum(r["ok"] for r in rows)
    print(f"\n{total}/{len(rows)} passed ({total / len(rows):.0%}) with {type(llm).__name__}")
    os.makedirs("results", exist_ok=True)
    json.dump({"model": type(llm).__name__, "passed": total, "total": len(rows), "rows": rows},
              open(f"results/eval_{'fake' if fake else 'groq'}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
