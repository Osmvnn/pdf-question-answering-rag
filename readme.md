# PDF Question Answering RAG

A learning project that answers questions using passages retrieved from an uploaded PDF. It uses Python, Streamlit, sentence-transformer embeddings, FAISS, and a local Ollama model.

## Current status — 27 September 2026

The PDF-to-Ollama pipeline is connected. Filename and physical PDF page numbers are preserved, and the app displays retrieved passages alongside generated answers with citations such as `[S1]`.

The current model is `llama3.2:3b`, with five retrieved passages. Answer reliability remains a work in progress: the model can produce unsupported claims even when its citation IDs are valid. Comparing a stronger model is deferred until the next session. A `llama3.1:8b` download was started and stopped before completion; it has not been evaluated or selected for the app. Ollama may retain partial download data outside this repository.

## How the pipeline works

1. **Load pages:** `extract_pages_from_pdf` returns text, filename, and a page number starting at 1. Blank pages do not renumber later pages.
2. **Split text:** `split_pages_into_chunks` creates overlapping passages of up to 700 characters with 200-character overlap. Chunks stay within one page so their source is unambiguous.
3. **Embed and index:** `all-MiniLM-L6-v2` embeds passage text. FAISS stores normalized vectors; the original chunk records retain metadata.
4. **Retrieve:** the question is embedded, FAISS finds similar passages, and the existing keyword boost reranks them. Scores are ranking heuristics, not answer confidence. Search is capped at the number of available chunks.
5. **Generate:** the top five passages receive IDs such as `S1` and are sent to `http://localhost:11434/api/chat`. The prompt requests at most two short sentences, evidence-only answers, citations, and acknowledgment of missing information. Temperature is 0.
6. **Review:** Streamlit displays the answer and expandable passages with filename, page, and scores. Code rejects missing or unknown citation IDs, but cannot prove that a cited passage supports a claim.

The prompt omits numeric bibliography markers such as `[2]` to reduce confusion with passage IDs. Original references remain in the stored and displayed passages. This convention fits the test guide; review it before using PDFs where bracketed numbers carry other meanings. The prompt also instructs the model not to infer authorship from a bibliography.

The old text-only loading, splitting, and answer-cleaning helpers remain available for notebook experiments. The app uses the metadata-aware pipeline.

## Project structure

```text
pdf-question-answering-rag/
|-- app/app.py                 # Streamlit interface
|-- src/
|   |-- answer_generator.py    # Ollama request and citation validation
|   |-- pdf_loader.py          # PDF text and page metadata
|   |-- text_splitter.py       # Overlapping chunks within pages
|   `-- vector_store.py        # FAISS indexing, search, and ranking
|-- data/                      # Local document storage
|-- notebooks/                 # Experiments
|-- tests/test_pipeline.py     # Offline regression checks
|-- test_rag.py                # Live retrieval and answer evaluation
|-- test_ollama.py             # Separate standalone connection test
|-- requirements.txt
`-- readme.md
```

## Setup and run

Run these commands in PowerShell from the project root. Skip creating the virtual environment if it already exists.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start Ollama and make sure `llama3.2:3b` is installed. If necessary:

```powershell
ollama pull llama3.2:3b
```

Then launch the app and upload a text-based PDF:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/app.py
```

The first embedding-model load may need internet access to download its files. Scanned PDFs need OCR, which is not implemented.

The app uses `requests` to call Ollama; it does not need the `ollama` Python package. The separate `test_ollama.py` uses that package and requires it in whichever Python environment runs that test. Its supplied-context connection check remains separate from PDF retrieval tests.

## Test step by step

The test guide is a two-page PDF that produces eight chunks with the current settings.

```powershell
$pdf = 'C:\Users\N01SC\Documents\Codex\2026-09-23\rw\outputs\Dental_Caries_Guide.pdf'

# 1. Check pipeline behavior without calling Ollama.
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v

# 2. Inspect retrieved passages before evaluating generated answers.
.\.venv\Scripts\python.exe -B test_rag.py $pdf --offline --retrieval-only

# 3. Generate answers to the six default questions.
.\.venv\Scripts\python.exe -B test_rag.py $pdf --offline

# 4. Compare three and five passages, saving each completed result.
.\.venv\Scripts\python.exe -B test_rag.py $pdf --offline --top-k 3 5 --output comparison.jsonl
```

`--offline` uses the cached embedding model; omit it if that model has not yet been downloaded. Use `--question "Your question"` to override the default questions; repeat the flag for multiple questions. Use a new output filename for each evaluation. Saved results include passages, answers or errors, generation time, and rejected raw answers when citation validation fails.

A successful script exit does not establish answer accuracy. Review every factual claim against its cited passage. For missing information, the expected response is:

> The supplied PDF passages do not provide this information.

This describes the retrieved evidence, not a guarantee that all information in the entire PDF was examined.

## Latest evaluation results

Manual review of one run per setting, using the same six questions and the final cleaned-context prompt with `llama3.2:3b`:

| Question | Three passages | Five passages |
| --- | --- | --- |
| How does dental caries develop? | Supported answer and citation | Supported answer and citation |
| How often should I brush my teeth, and what toothpaste should I use? | Supported answer and citation | Supported answer and citation |
| Can brushing regrow tooth structure already lost to a cavity? | Correct answer, wrong supporting passage cited | Supported answer and citation |
| What is the exact price of a filling in Dubai? | Correctly acknowledged missing information | Correctly acknowledged missing information |
| Who wrote this guide? | Invented an author from cited sources | Invented an author from cited sources |
| How often should I brush, and what is the exact price of a filling in Dubai? | Correct brushing advice and missing-price acknowledgment; unnecessary citation on the missing-information statement | Supported brushing advice and missing-price acknowledgment |

Five passages met the answer-and-support criteria on **5 of 6 questions**, compared with **4 of 6** for three passages. The app therefore retains five passages. These are small development-set results, not a general accuracy estimate; the prompt was adjusted using these questions.

Earlier runs copied the PDF's bibliography numbers as answer citations. Removing those markers from the model context improved citation formatting. It did not eliminate hallucination: the guide cites NIDCR/NHS material but does not name an author, and the model still attributed authorship to a cited institution.

All five offline regression tests passed. They cover chunk bounds and metadata, blank PDFs, fewer chunks than requested results, citation validation and connection errors, and empty evidence. Streamlit startup was also checked successfully during integration. Temporary evaluation outputs were removed after recording this summary.

## Next session: compare a stronger model

1. Choose and finish installing a stronger local model. `llama3.1:8b` was the proposed comparison candidate, not a verified improvement. The computer has about 16 GB of RAM and integrated Intel graphics, so measure response time as well as answer quality.
2. Hold the PDF, questions, prompt, and five-passage retrieval setting fixed. Change only the model with `--model`:

   ```powershell
   .\.venv\Scripts\python.exe -B test_rag.py $pdf --offline --top-k 5 --model llama3.1:8b --output comparison_8b.jsonl
   ```

3. Check every citation and require abstention on the author and price questions. Compare the results before changing the app's default model.
4. Add new questions and another PDF that were not used to tune the prompt, including partially answerable questions.

When diagnosing a failure, inspect retrieval first. If the necessary passage is absent, improve retrieval or chunking. If it is present but the answer misuses it, investigate generation. A valid citation label alone is never proof of a supported answer.
