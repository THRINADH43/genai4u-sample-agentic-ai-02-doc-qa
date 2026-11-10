"""Answer from retrieved chunks only, cite [source] after every claim, refuse otherwise."""

import json
import os
import re
import urllib.request

REFUSAL = "The documents do not say."
MIN_SCORE = 1.0

PROMPT = """Use ONLY the sources below. Write one or two short sentences and put the source id in
square brackets after each sentence, for example [refunds.md#1]. If the sources do not answer
the question, reply exactly: {refusal}

{sources}

Question: {question}"""


class FakeLLM:
    def complete(self, prompt):
        question = set(re.findall(r"[a-z0-9]{3,}", prompt.rsplit("Question:", 1)[1].lower()))
        best, overlap = None, 0
        for source, text in re.findall(r"^\[(\S+)\] (.*)$", prompt, re.M):
            for sentence in re.split(r"(?<=\.) ", text):
                o = len(question & set(re.findall(r"[a-z0-9]{3,}", sentence.lower())))
                if o > overlap:
                    best, overlap = f"{sentence} [{source}]", o
        return best or REFUSAL


class GroqLLM:
    def __init__(self, model="openai/gpt-oss-20b"):
        self.key, self.model = os.environ["GROQ_API_KEY"], model
        self.tokens = 0

    def complete(self, prompt):
        body = {"model": self.model, "temperature": 0, "reasoning_effort": "low",
                "messages": [{"role": "user", "content": prompt}]}
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                                              "User-Agent": "docqa/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.load(r)
        self.tokens += data["usage"]["total_tokens"]
        return data["choices"][0]["message"]["content"].strip()


def citations(answer):
    return re.findall(r"\[([^\]]+#\d+)\]", answer)


def valid_citations(answer, retrieved):
    """Every sentence has a citation, and every citation is one of the retrieved chunks."""
    if answer.strip() == REFUSAL:
        return True
    cited = citations(answer)
    sentences = [s for s in re.split(r"(?<=\])\s*", answer.strip()) if s.strip()]
    return bool(cited) and set(cited) <= retrieved and all(citations(s) for s in sentences)


def answer(question, index, llm, k=3):
    hits = [(s, c) for s, c in index.search(question, k) if s >= MIN_SCORE]
    if not hits:
        return REFUSAL, []
    sources = "\n".join(f"[{c.source}] {c.text}" for _, c in hits)
    reply = llm.complete(PROMPT.format(refusal=REFUSAL, sources=sources, question=question))
    if not valid_citations(reply, {c.source for _, c in hits}):
        return REFUSAL, hits
    return reply, hits
