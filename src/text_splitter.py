def split_pages_into_chunks(pages, chunk_size=700, overlap=200):
    """Split within pages so every chunk has one unambiguous source page."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size.")
    chunks = []
    for page in pages:
        text = page["text"].strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start + overlap + 1, end)
                if boundary != -1:
                    end = boundary
            passage = text[start:end].strip()
            if passage:
                chunks.append({"text": passage, "filename": page["filename"], "page": page["page"]})
            if end == len(text):
                break
            start = end - overlap
    return chunks


def split_text_by_paragraphs(text, chunk_size=700, overlap=200):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= chunk_size:
            current_chunk += paragraph + " "
        else:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:]
            current_chunk = overlap_text + " " + paragraph + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
