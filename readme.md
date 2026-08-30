# PDF Question Answering RAG

A lightweight Retrieval-Augmented Generation (RAG) application that lets you upload a PDF, search for the most relevant passages, and answer questions based on the content of the document.

This project uses a simple local pipeline:
- Extract text from the uploaded PDF
- Split the text into paragraph-based chunks
- Convert chunks into embeddings with a sentence-transformer model
- Store and search embeddings using FAISS
- Return the most relevant chunk and show a concise answer in the app

## Features

- Upload PDF files directly in the Streamlit UI
- Extract text from PDF pages using `pypdf`
- Chunk content with paragraph-based splitting and overlap
- Semantic retrieval using FAISS vector search
- Keyword-based ranking boost for better relevance
- View source passages and similarity scores
- Display a cleaned, short answer derived from the best matching chunk

## Tech Stack

- Python
- Streamlit
- `pypdf` for PDF parsing
- `sentence-transformers` for embeddings
- FAISS for vector similarity search
- NumPy for embedding processing

## Project Structure

```text
pdf-question-answering-rag/
├── app/
│   └── app.py                # Streamlit application entry point
├── src/
│   ├── answer_generator.py   # Answer cleaning logic
│   ├── pdf_loader.py         # PDF text extraction
│   ├── text_splitter.py      # Chunking logic
│   └── vector_store.py       # FAISS index creation and retrieval
├── data/                     # Document storage folder
├── notebooks/
│   └── rag_experiment.ipynb # Experiments and prototyping
├── readme.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── .gitignore                # Git ignore rules (if present)
```

## Installation

1. Open a terminal in the project root.
2. Create a virtual environment:

```bash
python -m venv .venv
```

3. Activate the environment:

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit app:

```bash
streamlit run app/app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How the App Works

1. The user uploads a PDF.
2. The app extracts the text from every page.
3. The text is split into smaller paragraph-based chunks.
4. Each chunk is converted into embeddings using the `all-MiniLM-L6-v2` model.
5. The vectors are stored in a FAISS index.
6. When a question is asked, the app embeds the question and retrieves the most relevant chunks.
7. The top matching chunk is cleaned and displayed as the answer.
8. The app also shows the source chunks and retrieval scores.

## Notes

- This is a lightweight prototype and not a production-grade LLM application.
- It works best with PDF files that contain readable text.
- The answer is generated from retrieved document chunks rather than from a separate LLM model.
- The project is suitable for learning and experimenting with basic RAG pipelines.

## Example Workflow

1. Upload a PDF containing company policies, course material, or technical documentation.
2. Ask a question such as: "What are the main benefits mentioned in the document?"
3. Review the retrieved source passages and the generated answer.

## Required Dependencies

The project depends on:
- `streamlit`
- `sentence-transformers`
- `faiss-cpu`
- `pypdf`
- `numpy`
- `torch`

For the complete list, see [requirements.txt](requirements.txt).

## License

This project is provided for educational and experimental use.
