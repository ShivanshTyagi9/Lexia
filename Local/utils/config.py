from dataclasses import dataclass

@dataclass
class RAGConfig:
    # Storage
    collection: str = "notes_papers"
    data_dir: str = "data"
    index_dir: str = "whoosh_index"

    # Models
    # embed_model: str = "intfloat/e5-small-v2"        # or "Alibaba-NLP/gte-small"
    # reranker_model: str = "BAAI/bge-reranker-base"  # or "-base" for more accuracy

    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"


    # Qdrant
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333
    dim: int = 384
    hnsw_M: int = 32
    hnsw_ef_construct: int = 256
    hnsw_ef_search: int = 64

    # Retrieval
    dense_top_k: int = 50
    bm25_top_k: int = 50
    rerank_top_k: int = 30
    final_top_k: int = 8

    # Chunking
    target_tokens_min: int = 400
    target_tokens_max: int = 800
    overlap_ratio: float = 0.18