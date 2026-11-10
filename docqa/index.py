"""Split documents into paragraph chunks and search them with BM25."""

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

STOP = {"a", "an", "the", "is", "are", "of", "in", "on", "to", "and", "or", "for", "do", "does",
        "can", "i", "my", "what", "how", "when", "which", "with", "be", "it", "if"}


@dataclass(frozen=True)
class Chunk:
    source: str   # e.g. "refunds.md#2"
    text: str


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def load_chunks(folder):
    """One chunk per paragraph; headings are kept with the paragraph after them."""
    chunks = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith((".md", ".txt")):
            continue
        text = open(os.path.join(folder, name), encoding="utf-8").read()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        n = 0
        heading = ""
        for p in paragraphs:
            if p.startswith("#") and "\n" not in p:
                heading = p.lstrip("# ") + ": "
                continue
            n += 1
            chunks.append(Chunk(f"{name}#{n}", heading + " ".join(p.split())))
    return chunks


class BM25Index:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c.text)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / len(self.len)
        self.df = Counter(w for t in self.tf for w in t)

    def search(self, query, k=3):
        n = len(self.chunks)
        scores = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for w in tokenize(query):
                if w in tf:
                    idf = math.log((n - self.df[w] + 0.5) / (self.df[w] + 0.5) + 1)
                    s += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            if s > 0:
                scores.append((s, self.chunks[i]))
        scores.sort(key=lambda sc: (-sc[0], sc[1].source))
        return scores[:k]
