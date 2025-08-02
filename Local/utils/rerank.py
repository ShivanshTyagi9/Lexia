from sentence_transformers import CrossEncoder
from typing import List

# Load BGE reranker model (base or large as per RAM)
reranker = CrossEncoder("BAAI/bge-reranker-base")


def rerank_chunks(query: str, docs: List, top_k=30, threshold=0.1):
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = [
        {"doc": doc, "score": score}
        for doc, score in zip(docs, scores)
        if score >= threshold
    ]

    # Sort and select top_k
    top_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)[:top_k]
    return [item["doc"] for item in top_docs]
