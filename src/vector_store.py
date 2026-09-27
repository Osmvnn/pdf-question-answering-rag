import numpy as np
import faiss


def build_faiss_index(chunks, embedding_model):
    if not chunks:
        raise ValueError("No readable text chunks to index.")
    texts = [chunk["text"] if isinstance(chunk, dict) else chunk for chunk in chunks]
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    embeddings_np = np.array(embeddings).astype("float32")

    dimension = embeddings_np.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)

    return index


def keyword_boost(question, retrieved_chunks):
    question_words = set(question.lower().replace("?", "").split())

    for item in retrieved_chunks:
        chunk_text = item["chunk"].lower()
        matches = sum(1 for word in question_words if word in chunk_text)
        item["score"] = item["score"] + (matches * 0.03)

    retrieved_chunks = sorted(
        retrieved_chunks,
        key=lambda x: x["score"],
        reverse=True
    )

    return retrieved_chunks


def retrieve_chunks(question, chunks, index, embedding_model, top_k=5):
    if not question.strip() or not chunks or top_k <= 0 or index.ntotal == 0:
        return []
    if index.ntotal != len(chunks):
        raise ValueError("The index and chunk list must describe the same document.")
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding_np = np.array(question_embedding).astype("float32")

    scores, indices = index.search(question_embedding_np, k=min(top_k, len(chunks)))

    retrieved = []

    for i, idx in enumerate(indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        retrieved.append({
            "chunk": chunk["text"] if isinstance(chunk, dict) else chunk,
            "filename": chunk.get("filename") if isinstance(chunk, dict) else None,
            "page": chunk.get("page") if isinstance(chunk, dict) else None,
            "similarity": float(scores[0][i]),
            "score": float(scores[0][i])
        })

    retrieved = keyword_boost(question, retrieved)
    for number, item in enumerate(retrieved, start=1):
        item["source_id"] = f"S{number}"

    return retrieved
