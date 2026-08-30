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