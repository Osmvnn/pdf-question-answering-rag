from pypdf import PdfReader
from pathlib import Path


def extract_pages_from_pdf(pdf, filename=None):
    """Keep physical PDF page numbers (1-based), including gaps for blank pages."""
    name = filename or (str(pdf) if isinstance(pdf, (str, Path)) else getattr(pdf, "name", "document.pdf"))
    return [
        {"text": text, "filename": Path(name).name, "page": number}
        for number, page in enumerate(PdfReader(pdf).pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]


def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text
