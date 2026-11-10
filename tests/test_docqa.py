import json
import os
import unittest

from docqa.index import BM25Index, load_chunks
from docqa.qa import REFUSAL, FakeLLM, answer, valid_citations
from evaluate import score

ROOT = os.path.join(os.path.dirname(__file__), "..")


class DocQATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = BM25Index(load_chunks(os.path.join(ROOT, "docs")))

    def test_chunks_have_ids_and_headings(self):
        chunks = load_chunks(os.path.join(ROOT, "docs"))
        self.assertIn("refunds.md#1", [c.source for c in chunks])
        self.assertTrue(chunks[0].text.startswith("Refund policy:"))

    def test_search_finds_the_right_paragraph(self):
        self.assertEqual(self.index.search("free shipping order")[0][1].source, "shipping.md#1")

    def test_answer_is_cited(self):
        reply, _ = answer("Does the warranty cover water damage?", self.index, FakeLLM())
        self.assertIn("[warranty.md#1]", reply)

    def test_refuses_when_documents_do_not_say(self):
        self.assertEqual(answer("Who is the founder?", self.index, FakeLLM())[0], REFUSAL)

    def test_made_up_citation_is_rejected(self):
        self.assertFalse(valid_citations("Free. [prices.md#9]", {"shipping.md#1"}))
        self.assertFalse(valid_citations("Free shipping above 499.", {"shipping.md#1"}))
        self.assertTrue(valid_citations("Free above Rs 499. [shipping.md#1]", {"shipping.md#1"}))

    def test_evaluation_runs_with_fake_model(self):
        questions = json.load(open(os.path.join(ROOT, "eval", "questions.json")))
        self.assertGreaterEqual(len(questions), 10)
        rows = score(questions, self.index, FakeLLM())
        self.assertEqual(len(rows), len(questions))


if __name__ == "__main__":
    unittest.main()
