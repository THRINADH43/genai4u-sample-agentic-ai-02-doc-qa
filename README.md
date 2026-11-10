# Document question answering with citations

A small RAG system over the shop policies in `docs/`. It chunks the documents, searches them,
answers only from the retrieved text, cites the source of every sentence, and says
"The documents do not say." when they don't. An evaluation set of 14 questions scores it.

## Setup

```
cp .env.example .env      # put your free Groq key in .env, or export GROQ_API_KEY
export GROQ_API_KEY=...   # never commit the key; .env is in .gitignore
python ask.py "Can I return opened software?"
python evaluate.py        # uses Groq if the key is set
python evaluate.py --fake # no key needed
python -m unittest discover -s tests -t .
```

Standard library only.

## How it works

- **Chunking** (`docqa/index.py`): one chunk per paragraph, with the section heading added
  in front, and an id like `refunds.md#2`.
- **Search:** BM25 over the chunks, top 3. Chunks scoring below 1.0 are dropped; if none are
  left, the answer is the refusal and the model is not called.
- **Answer** (`docqa/qa.py`): the prompt contains only the retrieved chunks with their ids.
  The model must put an id after each sentence or reply with the refusal.
- **Citation check in code:** every sentence must have a citation, and every citation must be
  one of the chunks retrieved for this question. A made-up citation (`[prices.md#9]`) or an
  uncited sentence turns the answer into the refusal.

## Evaluation

`eval/questions.json`: 14 questions, 11 answerable (with the words the answer must contain
and the source it must cite) and 3 that the documents do not answer (EMI, opening hours,
founder), which must be refused.

| model | passed |
|-------|--------|
| Groq `openai/gpt-oss-20b` | 13/14 (93%) |
| FakeLLM | 12/14 (86%) |

Saved runs: `results/eval_groq.json`, `results/eval_fake.json`.

**The failure:** q02 "How long does a refund take to reach me?" is refused. The answer is in
`refunds.md#2` ("Refunds go back to the original payment method within 7 working days"), but
search does not return that paragraph in the top 3. The top hit is `shipping.md#2`, because
"take" appears in "Other places take 4 to 7 days", and the other two refund paragraphs win on
"refund". The model then correctly refuses, because none of the retrieved text answers. Keyword
search misses questions that use different words from the document; embeddings or a bigger k
would fix this one. I kept the citation check strict, because a wrong answer about money is
worse than "the documents do not say".

## Tests

Chunk ids and headings, search, a cited answer, refusal, rejecting made-up or missing
citations, and the evaluation running end to end with the fake model.
