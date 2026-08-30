import numpy as np
import faiss


def build_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks, normalize_embeddings=True)
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
    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding_np = np.array(question_embedding).astype("float32")

    scores, indices = index.search(question_embedding_np, k=top_k)

    retrieved = []

    for i, idx in enumerate(indices[0]):
        retrieved.append({
            "chunk": chunks[idx],
            "score": scores[0][i]
        })

    retrieved = keyword_boost(question, retrieved)

    return retrieved