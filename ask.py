"""python ask.py "Can I return opened software?"   (Groq if GROQ_API_KEY is set, else the fake model)"""

import os
import sys

from docqa.index import BM25Index, load_chunks
from docqa.qa import FakeLLM, GroqLLM, answer

if __name__ == "__main__":
    llm = GroqLLM() if os.environ.get("GROQ_API_KEY") else FakeLLM()
    reply, hits = answer(" ".join(sys.argv[1:]), BM25Index(load_chunks("docs")), llm)
    print(reply)
