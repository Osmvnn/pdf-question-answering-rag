"""Live PDF retrieval + generation check, separate from test_ollama.py.

Usage: python test_rag.py path/to/document.pdf [--retrieval-only]
The default questions are specific to Dental_Caries_Guide.pdf.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

from sentence_transformers import SentenceTransformer

from src.pdf_loader import extract_pages_from_pdf
from src.text_splitter import split_pages_into_chunks
from src.vector_store import build_faiss_index, retrieve_chunks
from src.answer_generator import generate_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--question", action="append", help="Custom question; repeat for multiple questions")
    parser.add_argument("--top-k", type=int, nargs="+", default=[5], help="Passage counts to compare")
    parser.add_argument("--output", type=Path, help="Save results as JSON lines as each question completes")
    parser.add_argument("--offline", action="store_true", help="Use only the cached embedding model")
    parser.add_argument("--model", default="llama3.2:3b", help="Installed Ollama model to evaluate")
    args = parser.parse_args()
    if any(k <= 0 for k in args.top_k):
        parser.error("--top-k values must be positive")
    if args.output and args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve previous results")
    pages = extract_pages_from_pdf(args.pdf)
    chunks = split_pages_into_chunks(pages)
    model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=args.offline)
    index = build_faiss_index(chunks, model)
    print(f"Indexed {len(pages)} readable pages, {len(chunks)} chunks.", flush=True)
    questions = args.question or [
        "How does dental caries develop?",
        "How often should I brush my teeth, and what toothpaste should I use?",
        "Can brushing regrow tooth structure already lost to a cavity?",
        "What is the exact price of a filling in Dubai?",
        "Who wrote this guide?",
        "How often should I brush, and what is the exact price of a filling in Dubai?",
    ]
    for top_k in args.top_k:
        for question in questions:
            print(f"\nQUESTION: {question}", flush=True)
            print(f"TOP_K: {top_k}", flush=True)
            passages = retrieve_chunks(question, chunks, index, model, top_k=top_k)
            for p in passages:
                print(f"[{p['source_id']}] {p['filename']} page {p['page']} "
                      f"similarity={p['similarity']:.3f} rank={p['score']:.3f}\n{p['chunk']}\n", flush=True)
            result = {"question": question, "top_k": top_k, "model": args.model, "passages": passages}
            if not args.retrieval_only:
                started = perf_counter()
                try:
                    result["answer"] = generate_answer(question, passages, model=args.model)
                    print(f"ANSWER: {result['answer']}", flush=True)
                except RuntimeError as exc:
                    result["error"] = str(exc)
                    if hasattr(exc, "raw_answer"):
                        result["raw_answer"] = exc.raw_answer
                    print(f"ERROR: {exc}", flush=True)
                result["seconds"] = round(perf_counter() - started, 2)
            if args.output:
                with args.output.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
