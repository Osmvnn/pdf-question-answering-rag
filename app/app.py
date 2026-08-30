import sys
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.pdf_loader import extract_text_from_pdf
from src.text_splitter import split_text_by_paragraphs
from src.vector_store import build_faiss_index, retrieve_chunks
from src.answer_generator import clean_answer_text


st.set_page_config(
    page_title="PDF Question Answering RAG",
    page_icon="PDF",
    layout="wide"
)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def main():
    st.title("PDF Question Answering RAG")
    st.write(
        "Upload a PDF, ask a question, and retrieve the most relevant answer from the document."
    )

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        embedding_model = load_embedding_model()

        with st.spinner("Reading PDF..."):
            text = extract_text_from_pdf(uploaded_file)

        st.success("PDF loaded successfully.")
        st.write("Text length:", len(text))

        with st.spinner("Splitting and indexing document..."):
            chunks = split_text_by_paragraphs(
                text,
                chunk_size=700,
                overlap=200
            )
            index = build_faiss_index(chunks, embedding_model)

        st.write("Number of chunks:", len(chunks))

        question = st.text_input("Ask a question about the PDF")

        if question:
            retrieved_chunks = retrieve_chunks(
                question,
                chunks,
                index,
                embedding_model,
                top_k=5
            )

            best_answer = retrieved_chunks[0]["chunk"]
            clean_answer = clean_answer_text(best_answer, max_sentences=3)

            st.subheader("Generated Answer")
            st.info(clean_answer)

            st.subheader("Sources")
            for i, item in enumerate(retrieved_chunks, start=1):
                with st.expander(f"Source {i} | Score: {item['score']:.4f}"):
                    st.write(item["chunk"])


if __name__ == "__main__":
    main()