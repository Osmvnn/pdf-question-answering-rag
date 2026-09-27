import sys
from io import BytesIO
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pdf_loader import extract_pages_from_pdf
from src.text_splitter import split_pages_into_chunks
from src.vector_store import build_faiss_index, retrieve_chunks
from src.answer_generator import generate_answer


st.set_page_config(page_title="PDF Question Answering RAG", layout="wide")


@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def prepare_document(pdf_bytes, filename):
    pages = extract_pages_from_pdf(BytesIO(pdf_bytes), filename)
    chunks = split_pages_into_chunks(pages, chunk_size=700, overlap=200)
    return chunks, build_faiss_index(chunks, load_embedding_model())


def main():
    st.title("PDF Question Answering RAG")
    st.write("Upload a PDF and ask questions. Ollama answers using retrieved passages with citations.")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is None:
        return
    try:
        with st.spinner("Reading and indexing PDF..."):
            chunks, index = prepare_document(uploaded_file.getvalue(), uploaded_file.name)
    except Exception as exc:
        st.error(f"Could not index this PDF: {exc}")
        st.caption("Use a PDF containing selectable text. Scanned pages need OCR first.")
        return
    st.success(f"PDF loaded: {len(chunks)} chunks.")
    with st.form("question_form"):
        question = st.text_input("Ask a question about the PDF")
        submitted = st.form_submit_button("Generate answer")
    if not submitted:
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return
    with st.spinner("Retrieving supporting passages..."):
        passages = retrieve_chunks(question, chunks, index, load_embedding_model(), top_k=5)
    st.subheader("Generated answer")
    try:
        with st.spinner("Asking local Ollama (llama3.2:3b)..."):
            st.write(generate_answer(question, passages))
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
    st.subheader("Retrieved passages")
    st.caption("[S1], [S2], etc. refer to these passages. Page numbers count PDF pages from 1. "
               "Scores rank retrieval; they are not answer confidence. Check that citations support the claims.")
    for item in passages:
        with st.expander(f"[{item['source_id']}] {item['filename']} | Page {item['page']} | "
                         f"Ranking score: {item['score']:.3f}"):
            st.text(item["chunk"])
            st.caption(f"Cosine similarity before keyword boost: {item['similarity']:.3f}")


if __name__ == "__main__":
    main()
