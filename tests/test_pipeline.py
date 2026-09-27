import unittest
from io import BytesIO
from unittest.mock import patch, Mock

import numpy as np
import requests
from pypdf import PdfWriter

from src.pdf_loader import extract_pages_from_pdf
from src.text_splitter import split_pages_into_chunks
from src.vector_store import build_faiss_index, retrieve_chunks
from src.answer_generator import generate_answer, NO_EVIDENCE, AnswerValidationError


class FakeEmbeddings:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0] for _ in texts], dtype="float32")


class PipelineTests(unittest.TestCase):
    def test_chunk_boundaries_and_metadata(self):
        pages = [{"text": "word " * 300, "filename": "guide.pdf", "page": 2},
                 {"text": "next page", "filename": "guide.pdf", "page": 4}]
        chunks = split_pages_into_chunks(pages)
        self.assertTrue(all(0 < len(c["text"]) <= 700 for c in chunks))
        self.assertEqual(chunks[-1], pages[-1])
        self.assertTrue(all(c["page"] == 2 for c in chunks[:-1]))
        with self.assertRaises(ValueError):
            split_pages_into_chunks(pages, 200, 200)

    def test_blank_pdf(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        data = BytesIO()
        writer.write(data)
        data.seek(0)
        self.assertEqual(extract_pages_from_pdf(data, "blank.pdf"), [])
        with self.assertRaises(ValueError):
            build_faiss_index([], FakeEmbeddings())

    def test_fewer_chunks_than_top_k(self):
        chunks = [{"text": "fluoride", "filename": "guide.pdf", "page": 2}]
        index = build_faiss_index(chunks, FakeEmbeddings())
        results = retrieve_chunks("fluoride?", chunks, index, FakeEmbeddings(), top_k=5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source_id"], "S1")
        self.assertEqual(results[0]["page"], 2)
        self.assertEqual(results[0]["filename"], "guide.pdf")

    @patch("src.answer_generator.requests.post")
    def test_generation_contract(self, post):
        passages = [{"source_id": "S1", "chunk": "Brush twice daily. [3]", "filename": "guide.pdf", "page": 2}]
        response = Mock()
        post.return_value = response
        response.json.return_value = {"message": {"content": "Brush twice daily. [S1]"}}
        self.assertEqual(generate_answer("How often?", passages), "Brush twice daily. [S1]")
        payload = post.call_args.kwargs["json"]
        self.assertIn("Brush twice daily.", payload["messages"][1]["content"])
        self.assertNotIn("[3]", payload["messages"][1]["content"])
        self.assertEqual(passages[0]["chunk"], "Brush twice daily. [3]")
        for answer in ["Unsupported answer", "Answer [S9]", ""]:
            response.json.return_value = {"message": {"content": answer}}
            with self.assertRaises(AnswerValidationError) as caught:
                generate_answer("Question", passages)
            self.assertEqual(caught.exception.raw_answer, answer)
        response.json.return_value = {"message": {"content": NO_EVIDENCE}}
        self.assertEqual(generate_answer("Price?", passages), NO_EVIDENCE)
        post.side_effect = requests.ConnectionError("offline")
        with self.assertRaises(RuntimeError):
            generate_answer("Question", passages)

    @patch("src.answer_generator.requests.post")
    def test_no_evidence_skips_model(self, post):
        self.assertEqual(generate_answer("Question", []), NO_EVIDENCE)
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
