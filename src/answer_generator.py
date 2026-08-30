def clean_answer_text(text, max_sentences=3):
    text = text.replace("\n", " ")
    sentences = text.split(". ")

    cleaned = ". ".join(sentences[:max_sentences])

    if not cleaned.endswith("."):
        cleaned += "."

    return cleaned